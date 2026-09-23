#!/usr/bin/env python3
import argparse
import json
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def multipart_body(image_path):
    boundary = f"----resnet18-{uuid.uuid4().hex}"
    image = Path(image_path).read_bytes()
    filename = Path(image_path).name
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        "Content-Type: image/jpeg\r\n\r\n"
    ).encode() + image + f"\r\n--{boundary}--\r\n".encode()
    content_type = f"multipart/form-data; boundary={boundary}"
    return body, content_type


def send_request(url, body, content_type, timeout):
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": content_type},
        method="POST",
    )
    started_at = time.perf_counter()

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read())
        latency_ms = (time.perf_counter() - started_at) * 1000
        return latency_ms, payload.get("timings_ms", {}), None
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        latency_ms = (time.perf_counter() - started_at) * 1000
        return latency_ms, {}, str(error)


def percentile(values, percent):
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * percent / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def print_stats(name, values):
    mean = sum(values) / len(values)
    print(
        f"{name:<20} mean={mean:8.3f} ms  "
        f"p50={percentile(values, 50):8.3f} ms  "
        f"p95={percentile(values, 95):8.3f} ms  "
        f"p99={percentile(values, 99):8.3f} ms"
    )


def main():
    parser = argparse.ArgumentParser(description="Load test the ResNet18 API.")
    parser.add_argument("--url", default="http://localhost:8001/infer")
    parser.add_argument("--image", required=True)
    parser.add_argument("-n", "--requests", type=int, default=100)
    parser.add_argument("-c", "--concurrency", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=30)
    args = parser.parse_args()

    if args.requests < 1 or args.concurrency < 1 or args.warmup < 0:
        parser.error("requests/concurrency must be positive and warmup non-negative")

    body, content_type = multipart_body(args.image)

    print(f"Warming up with {args.warmup} requests...")
    for _ in range(args.warmup):
        _, _, error = send_request(args.url, body, content_type, args.timeout)
        if error:
            raise SystemExit(f"Warmup failed: {error}")

    print(
        f"Sending {args.requests} requests "
        f"with concurrency={args.concurrency}..."
    )
    started_at = time.perf_counter()
    results = []

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(
                send_request,
                args.url,
                body,
                content_type,
                args.timeout,
            )
            for _ in range(args.requests)
        ]
        for future in as_completed(futures):
            results.append(future.result())

    wall_time = time.perf_counter() - started_at
    successful = [(latency, timings) for latency, timings, error in results if not error]
    errors = [error for _, _, error in results if error]

    print(
        f"\nCompleted: {len(successful)}/{args.requests}, "
        f"errors: {len(errors)}, throughput: {len(successful) / wall_time:.2f} req/s"
    )

    if not successful:
        raise SystemExit(f"All requests failed. First error: {errors[0]}")

    print("\nClient request latency:")
    print_stats("request", [latency for latency, _ in successful])

    print("\nServer component latency:")
    component_names = (
        "preprocessing",
        "h2d",
        "tensorrt_gpu",
        "d2h",
        "postprocessing",
    )
    component_means = {}
    for component in component_names:
        values = [
            timings[component]
            for _, timings in successful
            if component in timings
        ]
        if values:
            print_stats(component, values)
            component_means[component] = sum(values) / len(values)

    request_mean = sum(latency for latency, _ in successful) / len(successful)
    measured_mean = sum(component_means.values())
    other_mean = max(0.0, request_mean - measured_mean)

    print("\nMean latency breakdown:")
    print(f"request latency      {request_mean:8.3f} ms")
    labels = {
        "preprocessing": "preprocessing",
        "h2d": "H2D",
        "tensorrt_gpu": "TensorRT GPU",
        "d2h": "D2H",
        "postprocessing": "postprocessing",
    }
    for component in component_names:
        value = component_means.get(component, 0.0)
        print(f"├── {labels[component]:<16} {value:8.3f} ms")
    print(f"└── other/queue       {other_mean:8.3f} ms")

    if errors:
        print(f"\nFirst error: {errors[0]}")


if __name__ == "__main__":
    main()
