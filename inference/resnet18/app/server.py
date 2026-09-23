from time import perf_counter

from fastapi import FastAPI, File, Request, UploadFile
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from starlette.concurrency import run_in_threadpool
from starlette.responses import Response
import uvicorn

import infer

app = FastAPI()

INFERENCE_REQUESTS = Counter(
    "resnet18_inference_requests_total",
    "Total number of inference requests.",
    ["status"],
)
REQUEST_DURATION = Histogram(
    "resnet18_request_duration_seconds",
    "End-to-end duration of the /infer HTTP request.",
    buckets=(
        0.001,
        0.0025,
        0.005,
        0.0075,
        0.01,
        0.015,
        0.02,
        0.03,
        0.05,
        0.075,
        0.1,
        0.15,
        0.25,
        0.5,
        1.0,
        2.5,
    ),
)
COMPONENT_DURATION = Histogram(
    "resnet18_inference_component_duration_seconds",
    "Duration of an inference pipeline component.",
    ["component"],
    buckets=(
        0.00005,
        0.0001,
        0.00025,
        0.0005,
        0.001,
        0.0025,
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
    ),
)
UPLOAD_SIZE = Histogram(
    "resnet18_upload_bytes",
    "Uploaded image size in bytes.",
    buckets=(10_000, 50_000, 100_000, 500_000, 1_000_000, 5_000_000),
)
INFERENCES_IN_PROGRESS = Gauge(
    "resnet18_inferences_in_progress",
    "Number of inference requests currently being processed.",
)


@app.middleware("http")
async def measure_request_latency(request: Request, call_next):
    if request.url.path != "/infer":
        return await call_next(request)

    started_at = perf_counter()
    try:
        return await call_next(request)
    finally:
        REQUEST_DURATION.observe(perf_counter() - started_at)


@app.post("/infer")
async def predict(file: UploadFile = File(...)):
    image = await file.read()
    UPLOAD_SIZE.observe(len(image))
    INFERENCES_IN_PROGRESS.inc()

    try:
        result, timings_ms = await run_in_threadpool(
            infer.predict_with_timings,
            image,
        )
        for component, duration_ms in timings_ms.items():
            COMPONENT_DURATION.labels(component=component).observe(
                duration_ms / 1000
            )
        INFERENCE_REQUESTS.labels(status="success").inc()
        return {"result": result, "timings_ms": timings_ms}
    except Exception:
        INFERENCE_REQUESTS.labels(status="error").inc()
        raise
    finally:
        INFERENCES_IN_PROGRESS.dec()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    infer.get_predictor()
    uvicorn.run(app, host="0.0.0.0", port=8001)
