import argparse
import io
import os
import threading
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image
import tensorrt as trt
from cuda.bindings import runtime as cudart


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def check_cuda(result):
    err = result[0]

    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA error: {err}")

    if len(result) == 1:
        return None

    return result[1]


def resolve_engine_path():
    env_path = os.environ.get("ENGINE_PATH")
    if env_path:
        return env_path

    here = Path(__file__).resolve().parent
    for candidate in (
        here / "resnet18_fp16.engine",
        here.parent / "resnet18_fp16.engine",
    ):
        if candidate.is_file():
            return str(candidate)

    return str(here / "resnet18_fp16.engine")


DEFAULT_ENGINE = resolve_engine_path()


def open_image(image):
    if isinstance(image, Image.Image):
        pil_image = image
    elif isinstance(image, (bytes, bytearray)):
        pil_image = Image.open(io.BytesIO(image))
    else:
        pil_image = Image.open(image)

    return pil_image.convert("RGB")


def preprocess_image(image):
    image = open_image(image)

    # Standard preprocessing for torchvision ResNet18:
    # resize shorter side -> 256, then center crop 224x224
    width, height = image.size

    if width < height:
        new_width = 256
        new_height = round(height * 256 / width)
    else:
        new_height = 256
        new_width = round(width * 256 / height)

    image = image.resize((new_width, new_height), Image.BILINEAR)

    left = (new_width - 224) // 2
    top = (new_height - 224) // 2

    image = image.crop((left, top, left + 224, top + 224))

    # HWC uint8 -> float32 [0, 1]
    array = np.asarray(image, dtype=np.float32) / 255.0

    # ImageNet normalization
    array = (array - IMAGENET_MEAN) / IMAGENET_STD

    # HWC -> CHW
    array = np.transpose(array, (2, 0, 1))

    # Add batch dimension: CHW -> NCHW
    array = np.expand_dims(array, axis=0)

    return np.ascontiguousarray(array, dtype=np.float32)


def softmax(x):
    x = x - np.max(x)
    exp = np.exp(x)
    return exp / np.sum(exp)


def load_engine(engine_path):
    with open(engine_path, "rb") as f:
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(f.read())

    if engine is None:
        raise RuntimeError("Failed to deserialize TensorRT engine")

    return engine


class Predictor:
    def __init__(self, engine_path=DEFAULT_ENGINE):
        self.engine = load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.input_name = None
        self.output_name = None

        print("TensorRT tensors:")

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            shape = self.engine.get_tensor_shape(name)
            dtype = self.engine.get_tensor_dtype(name)

            print(f"  {name}: mode={mode}, shape={shape}, dtype={dtype}")

            if mode == trt.TensorIOMode.INPUT:
                self.input_name = name
            elif mode == trt.TensorIOMode.OUTPUT:
                self.output_name = name

        if self.input_name is None or self.output_name is None:
            raise RuntimeError("Could not find input/output tensors")

        self._lock = threading.Lock()

    def predict(self, image):
        results, _ = self.predict_with_timings(image)
        return results

    def predict_with_timings(self, image):
        with self._lock:
            return self._predict(image)

    def _predict(self, image):
        preprocessing_started = perf_counter()
        host_input = preprocess_image(image)
        preprocessing_ms = (perf_counter() - preprocessing_started) * 1000

        engine_input_shape = tuple(self.engine.get_tensor_shape(self.input_name))

        if -1 in engine_input_shape:
            self.context.set_input_shape(self.input_name, host_input.shape)

        output_shape = tuple(self.context.get_tensor_shape(self.output_name))
        output_dtype = trt.nptype(self.engine.get_tensor_dtype(self.output_name))
        host_output = np.empty(output_shape, dtype=output_dtype)

        device_input = check_cuda(cudart.cudaMalloc(host_input.nbytes))
        device_output = check_cuda(cudart.cudaMalloc(host_output.nbytes))
        stream = check_cuda(cudart.cudaStreamCreate())
        events = [check_cuda(cudart.cudaEventCreate()) for _ in range(6)]
        h2d_start, h2d_end, gpu_start, gpu_end, d2h_start, d2h_end = events

        try:
            check_cuda(cudart.cudaEventRecord(h2d_start, stream))
            check_cuda(
                cudart.cudaMemcpyAsync(
                    device_input,
                    host_input.ctypes.data,
                    host_input.nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                    stream,
                )
            )
            check_cuda(cudart.cudaEventRecord(h2d_end, stream))

            if not self.context.set_tensor_address(self.input_name, int(device_input)):
                raise RuntimeError("Failed to bind input tensor")

            if not self.context.set_tensor_address(self.output_name, int(device_output)):
                raise RuntimeError("Failed to bind output tensor")

            check_cuda(cudart.cudaEventRecord(gpu_start, stream))
            if not self.context.execute_async_v3(stream):
                raise RuntimeError("TensorRT inference failed")
            check_cuda(cudart.cudaEventRecord(gpu_end, stream))

            check_cuda(cudart.cudaEventRecord(d2h_start, stream))
            check_cuda(
                cudart.cudaMemcpyAsync(
                    host_output.ctypes.data,
                    device_output,
                    host_output.nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                    stream,
                )
            )
            check_cuda(cudart.cudaEventRecord(d2h_end, stream))

            check_cuda(cudart.cudaStreamSynchronize(stream))

            h2d_ms = check_cuda(cudart.cudaEventElapsedTime(h2d_start, h2d_end))
            gpu_ms = check_cuda(cudart.cudaEventElapsedTime(gpu_start, gpu_end))
            d2h_ms = check_cuda(cudart.cudaEventElapsedTime(d2h_start, d2h_end))
        finally:
            for event in events:
                cudart.cudaEventDestroy(event)
            cudart.cudaFree(device_input)
            cudart.cudaFree(device_output)
            cudart.cudaStreamDestroy(stream)

        postprocessing_started = perf_counter()
        logits = host_output.reshape(-1)
        probabilities = softmax(logits)
        top5 = np.argsort(probabilities)[-5:][::-1]

        results = [
            {
                "rank": rank,
                "class_id": int(class_id),
                "probability": float(probabilities[class_id]),
            }
            for rank, class_id in enumerate(top5, start=1)
        ]
        postprocessing_ms = (perf_counter() - postprocessing_started) * 1000

        timings_ms = {
            "preprocessing": preprocessing_ms,
            "h2d": h2d_ms,
            "tensorrt_gpu": gpu_ms,
            "d2h": d2h_ms,
            "postprocessing": postprocessing_ms,
        }

        return results, timings_ms


_predictor = None


def get_predictor(engine_path=DEFAULT_ENGINE):
    global _predictor

    if _predictor is None:
        _predictor = Predictor(engine_path)

    return _predictor


def predict(image, engine_path=DEFAULT_ENGINE):
    return get_predictor(engine_path).predict(image)


def predict_with_timings(image, engine_path=DEFAULT_ENGINE):
    return get_predictor(engine_path).predict_with_timings(image)


def infer(engine_path, image_path):
    results = predict(image_path, engine_path=engine_path)

    print("\nTop-5 predictions:")

    for item in results:
        print(
            f"{item['rank']}. class_id={item['class_id']:<4} "
            f"probability={item['probability']:.4f}"
        )

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "image",
        help="Path to input image"
    )

    parser.add_argument(
        "--engine",
        default=DEFAULT_ENGINE,
        help="Path to TensorRT engine"
    )

    args = parser.parse_args()

    infer(args.engine, args.image)