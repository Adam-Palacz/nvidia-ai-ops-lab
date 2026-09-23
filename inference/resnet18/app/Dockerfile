# syntax=docker/dockerfile:1
#
# Build from inference/resnet18 (not from app/):
#
#   sudo docker build -f app/Dockerfile \
#     --build-context trt_py=/usr/lib/python3.12/dist-packages \
#     --build-context trt_lib=/usr/lib/aarch64-linux-gnu \
#     -t resnet18:1.0 .
#
# Run with the Jetson GPU:
#
#   sudo docker run --rm --runtime nvidia -p 8001:8001 resnet18:1.0

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    ENGINE_PATH=/app/resnet18_fp16.engine \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY app/requirements.txt /app/requirements.txt
RUN pip3 install --break-system-packages --no-cache-dir -r /app/requirements.txt

# TensorRT 10.16 from this Jetson. Do not pip install tensorrt (that pulls 11.3).
COPY --from=trt_py tensorrt /usr/lib/python3.12/dist-packages/tensorrt
COPY --from=trt_py tensorrt-10.16.2.10.dist-info /usr/lib/python3.12/dist-packages/tensorrt-10.16.2.10.dist-info
COPY --from=trt_lib libnvinfer.so.10.16.2 /usr/lib/aarch64-linux-gnu/
COPY --from=trt_lib libnvinfer_plugin.so.10.16.2 /usr/lib/aarch64-linux-gnu/
COPY --from=trt_lib libnvinfer_vc_plugin.so.10.16.2 /usr/lib/aarch64-linux-gnu/
COPY --from=trt_lib libnvonnxparser.so.10.16.2 /usr/lib/aarch64-linux-gnu/

RUN ln -sf libnvinfer.so.10.16.2 /usr/lib/aarch64-linux-gnu/libnvinfer.so.10 \
    && ln -sf libnvinfer_plugin.so.10.16.2 /usr/lib/aarch64-linux-gnu/libnvinfer_plugin.so.10 \
    && ln -sf libnvinfer_vc_plugin.so.10.16.2 /usr/lib/aarch64-linux-gnu/libnvinfer_vc_plugin.so.10 \
    && ln -sf libnvonnxparser.so.10.16.2 /usr/lib/aarch64-linux-gnu/libnvonnxparser.so.10 \
    && ldconfig

COPY app/infer.py app/server.py /app/
COPY resnet18_fp16.engine /app/resnet18_fp16.engine

EXPOSE 8001

CMD ["python3", "server.py"]
