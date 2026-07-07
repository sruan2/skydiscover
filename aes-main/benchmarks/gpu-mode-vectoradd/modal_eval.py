"""Deployable Modal H100 evaluator for the GPU MODE VectorAdd benchmark.

Deploy once from the repository root:

    uv run --extra gpu-mode-vectoradd modal deploy benchmarks/gpu-mode-vectoradd/modal_eval.py
"""

from __future__ import annotations

import modal


TEST_CASES = (
    {"N": 256, "seed": 42},
    {"N": 512, "seed": 123},
    {"N": 1024, "seed": 456},
    {"N": 2048, "seed": 789},
)
BENCHMARK_CASES = (
    {"N": 1024, "seed": 1001},
    {"N": 2048, "seed": 1002},
    {"N": 4096, "seed": 1003},
    {"N": 8192, "seed": 1004},
)

SCORE_SCALE = 3000.0
MAX_REPEATS = 100
RELATIVE_ERROR_TARGET = 0.001
MAX_CASE_TIME_NS = 10e9
MAX_WALL_TIME_NS = 120e9

image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel",
        add_python="3.11",
    )
    .pip_install("triton")
)
app = modal.App("aes-gpu-mode-vectoradd-eval")


@app.function(gpu="H100", image=image, timeout=300)
def evaluate_kernel(kernel_code: str, mode: str = "leaderboard") -> str:
    import copy
    import gc
    import importlib.util
    import json
    import math
    import os
    import tempfile
    import time
    import traceback

    import torch

    def generate_input(N: int, seed: int):
        generator = torch.Generator(device="cuda")
        generator.manual_seed(seed)
        a = torch.randn((N, N), device="cuda", dtype=torch.float16, generator=generator)
        b = torch.randn((N, N), device="cuda", dtype=torch.float16, generator=generator)
        return a, b

    def clone_data(data):
        if isinstance(data, tuple):
            return tuple(clone_data(item) for item in data)
        if isinstance(data, torch.Tensor):
            return data.clone()
        return copy.deepcopy(data)

    def check(data, output):
        expected = data[0] + data[1]
        if not isinstance(output, torch.Tensor):
            return False, f"expected Tensor output, got {type(output).__name__}"
        if output.shape != expected.shape:
            return False, f"shape mismatch: expected {expected.shape}, got {output.shape}"
        if output.device.type != "cuda":
            return False, f"device mismatch: expected cuda, got {output.device}"
        if output.dtype != torch.float16:
            return False, f"dtype mismatch: expected float16, got {output.dtype}"
        if torch.allclose(output, expected, rtol=1e-3, atol=1e-3):
            return True, ""
        difference = torch.abs(output.float() - expected.float())
        return False, f"output mismatch: max_diff={difference.max().item():.6f}"

    def stats(samples):
        count = len(samples)
        mean = sum(samples) / count
        if count == 1:
            return {"runs": count, "mean": mean, "std": 0.0, "err": 0.0}
        variance = sum((sample - mean) ** 2 for sample in samples) / (count - 1)
        std = math.sqrt(variance)
        return {"runs": count, "mean": mean, "std": std, "err": std / math.sqrt(count)}

    # Two times H100's 50 MB L2 cache. Touching it before every timed call makes
    # measurements consistently exercise HBM rather than an accidentally warm L2.
    l2_flush = torch.zeros(100 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")

    def flush_l2():
        l2_flush.zero_()
        torch.cuda.synchronize()

    def benchmark_case(kernel, arguments, max_time_ns=MAX_CASE_TIME_NS):
        data = generate_input(**arguments)
        original = clone_data(data)
        output = kernel(data)
        torch.cuda.synchronize()
        passed, message = check(original, output)
        if not passed:
            return None, f"benchmark correctness failed: {message}"
        del output

        samples = []
        wall_start = time.perf_counter_ns()
        for _ in range(MAX_REPEATS):
            flush_l2()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = kernel(data)
            end.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(end) * 1e6)  # milliseconds to nanoseconds
            del output

            if len(samples) > 2:
                summary = stats(samples)
                if summary["mean"] and summary["err"] / summary["mean"] < RELATIVE_ERROR_TARGET:
                    break
                if summary["mean"] * summary["runs"] > max_time_ns:
                    break
                if time.perf_counter_ns() - wall_start > MAX_WALL_TIME_NS:
                    break
        return stats(samples), None

    gpu_name = torch.cuda.get_device_name(0)
    common = {
        "gpu_name": gpu_name,
        "torch_version": torch.__version__,
        "platform": "modal-h100",
    }

    temporary_directory = tempfile.mkdtemp(prefix="aes-gpu-mode-vectoradd-")
    submission_path = os.path.join(temporary_directory, "submission.py")
    with open(submission_path, "w") as handle:
        handle.write(kernel_code)

    try:
        spec = importlib.util.spec_from_file_location("submission", submission_path)
        if spec is None or spec.loader is None:
            raise RuntimeError("could not create submission module spec")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        kernel = module.custom_kernel
    except Exception:
        return json.dumps(
            {
                **common,
                "success": False,
                "tests_passed": 0,
                "tests_total": len(TEST_CASES),
                "test_details": [],
                "failure_stage": "import",
                "error": traceback.format_exc(),
            }
        )

    test_details = []
    passed_count = 0
    for test_case in TEST_CASES:
        try:
            data = generate_input(**test_case)
            original = clone_data(data)
            output = kernel(data)
            torch.cuda.synchronize()
            passed, message = check(original, output)
        except Exception:
            passed = False
            message = traceback.format_exc()[:1000]
        test_details.append({**test_case, "passed": passed, "error": message})
        passed_count += int(passed)

    if passed_count != len(TEST_CASES):
        return json.dumps(
            {
                **common,
                "success": False,
                "tests_passed": passed_count,
                "tests_total": len(TEST_CASES),
                "test_details": test_details,
                "failure_stage": "correctness",
                "error": "correctness check failed",
            }
        )

    if mode == "test":
        return json.dumps(
            {
                **common,
                "success": True,
                "tests_passed": passed_count,
                "tests_total": len(TEST_CASES),
                "test_details": test_details,
            }
        )

    gc.collect()
    torch.cuda.empty_cache()
    benchmark_case(kernel, BENCHMARK_CASES[0], max_time_ns=1e8)

    benchmark_details = []
    means_ns = []
    for arguments in BENCHMARK_CASES:
        summary, error = benchmark_case(kernel, arguments)
        if error:
            return json.dumps(
                {
                    **common,
                    "success": False,
                    "tests_passed": passed_count,
                    "tests_total": len(TEST_CASES),
                    "test_details": test_details,
                    "failure_stage": "benchmark",
                    "error": error,
                }
            )
        mean_us = summary["mean"] / 1e3
        err_us = summary["err"] / 1e3
        benchmark_details.append(
            {
                **arguments,
                "mean_us": round(mean_us, 3),
                "err_us": round(err_us, 3),
                "runs": summary["runs"],
            }
        )
        means_ns.append(summary["mean"])

    geomean_us = math.prod(means_ns) ** (1.0 / len(means_ns)) / 1e3
    return json.dumps(
        {
            **common,
            "success": True,
            "tests_passed": passed_count,
            "tests_total": len(TEST_CASES),
            "test_details": test_details,
            "benchmark": {
                "geomean_us": round(geomean_us, 3),
                "score": round(SCORE_SCALE / geomean_us, 3),
            },
            "benchmark_details": benchmark_details,
        }
    )
