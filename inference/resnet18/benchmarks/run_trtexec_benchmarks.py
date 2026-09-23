#!/usr/bin/env python3
import argparse
import json
import re
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path


SUMMARY_PATTERNS = {
    "throughput_qps": r"Throughput: ([0-9.]+) qps",
    "latency_mean_ms": r"Latency: .*?mean = ([0-9.]+) ms",
    "latency_median_ms": r"Latency: .*?median = ([0-9.]+) ms",
    "latency_p95_ms": r"Latency: .*?percentile\(95%\) = ([0-9.]+) ms",
    "enqueue_mean_ms": r"Enqueue Time: .*?mean = ([0-9.]+) ms",
    "gpu_compute_mean_ms": r"GPU Compute Time: .*?mean = ([0-9.]+) ms",
    "h2d_mean_ms": r"H2D Latency: .*?mean = ([0-9.]+) ms",
    "d2h_mean_ms": r"D2H Latency: .*?mean = ([0-9.]+) ms",
    "host_walltime_s": r"Total Host Walltime: ([0-9.]+) s",
    "gpu_compute_total_s": r"Total GPU Compute Time: ([0-9.]+) s",
}


def parse_summary(output):
    result = {}
    for name, pattern in SUMMARY_PATTERNS.items():
        match = re.search(pattern, output)
        if not match:
            raise RuntimeError(f"Could not parse {name} from trtexec output")
        result[name] = float(match.group(1))
    return result


def read_process_output(process):
    output, _ = process.communicate(timeout=5)
    return output


def run_benchmark(engine, precision, streams, warmup_ms, duration_s):
    command = [
        "trtexec",
        f"--loadEngine={engine}",
        f"--warmUp={warmup_ms}",
        f"--duration={duration_s}",
        f"--infStreams={streams}",
        "--useSpinWait",
    ]

    tegrastats = subprocess.Popen(
        ["tegrastats", "--interval", "100"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    finally:
        tegrastats.terminate()

    tegrastats_output = read_process_output(tegrastats)
    gpu_samples = [
        int(value)
        for value in re.findall(r"GR3D_FREQ ([0-9]+)%", tegrastats_output)
    ]

    result = {
        "precision": precision,
        "streams": streams,
        "command": " ".join(command),
        **parse_summary(completed.stdout + completed.stderr),
    }

    if gpu_samples:
        result.update(
            {
                "gpu_util_mean_percent": round(statistics.mean(gpu_samples), 1),
                "gpu_util_max_percent": max(gpu_samples),
                "gpu_util_samples": len(gpu_samples),
            }
        )

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup-ms", type=int, default=2000)
    parser.add_argument("--duration", type=int, default=10)
    parser.add_argument(
        "--output",
        default="benchmarks/results/phase1.json",
    )
    args = parser.parse_args()

    engines = (
        ("FP32", "resnet18_fp32.engine"),
        ("FP16", "resnet18_fp16.engine"),
    )
    streams = (1, 2, 4, 8)
    results = []

    for precision, engine in engines:
        if not Path(engine).is_file():
            raise SystemExit(
                f"Missing {engine}. Run scripts/build_engines.sh first."
            )
        for stream_count in streams:
            print(f"Running {precision}, streams={stream_count}...", flush=True)
            result = run_benchmark(
                engine,
                precision,
                stream_count,
                args.warmup_ms,
                args.duration,
            )
            results.append(result)
            print(
                f"  {result['throughput_qps']:.1f} qps, "
                f"GPU {result.get('gpu_util_mean_percent', 0):.1f}%, "
                f"compute {result['gpu_compute_mean_ms']:.3f} ms"
            )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "platform": {
            "device": "NVIDIA Jetson Orin Nano Super Developer Kit",
            "power_mode": "25W",
            "jetson_linux": "R39.2.1",
            "tensorrt": "10.16.2",
            "nsight_systems": "2025.6.3",
        },
        "methodology": {
            "warmup_ms": args.warmup_ms,
            "duration_s": args.duration,
            "data_transfers": True,
            "spin_wait": True,
            "cuda_graph": False,
            "clocks_locked": False,
        },
        "results": results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
