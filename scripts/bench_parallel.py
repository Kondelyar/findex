"""Lab 5 benchmark: serial vs threads vs processes on one corpus.

Run it with the interpreter you want to measure; every measurement is a fresh
subprocess, so peak memory belongs to that one run and nothing is warm.

    uv run python scripts/bench_parallel.py data/big.jsonl --out bench/gil.json
    UV_PROJECT_ENVIRONMENT=.venv-313t uv run --python 3.13t --no-dev \
        python scripts/bench_parallel.py data/big.jsonl --out bench/nogil.json --executors threads

Then: uv run --group bench python scripts/bench_report.py bench/gil.json bench/nogil.json
"""

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

from findex.corpus import corpus_size, plan_chunks
from findex.parallel import ExecutorKind, gc_paused, merge, run_chunks


def peak_rss_mb() -> float | None:
    if sys.platform == "win32":
        try:
            import psutil
        except ImportError:
            return None
        return psutil.Process().memory_info().peak_wset / 2**20
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (2**20 if sys.platform == "darwin" else 2**10)


def children_cpu() -> float | None:
    if sys.platform == "win32":
        return None
    import resource

    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return usage.ru_utime + usage.ru_stime


def run_once(corpus: Path, kind: ExecutorKind, workers: int, positions: bool) -> dict:
    cpu_start = time.process_time()
    start = time.perf_counter()
    chunks = plan_chunks(corpus, workers)
    planned = time.perf_counter()
    with gc_paused():
        results = list(run_chunks(chunks, workers, kind, positions))
        mapped = time.perf_counter()
        index = merge(result.partial for result in results)
        done = time.perf_counter()

    cpu = time.process_time() - cpu_start
    if kind is ExecutorKind.processes:
        # exact on Unix (includes startup and pickling the result); on Windows
        # fall back to what each worker reported when its last task ended
        exact = children_cpu()
        if exact is not None:
            cpu += exact
        else:
            by_pid: dict[int, float] = {}
            for result in results:
                by_pid[result.pid] = max(by_pid.get(result.pid, 0.0), result.cpu)
            cpu += sum(by_pid.values())

    return {
        "wall": done - start,
        "plan": planned - start,
        "map": mapped - planned,
        "merge": done - mapped,
        "cpu": cpu,
        "rss_mb": peak_rss_mb(),
        "docs": index.num_docs,
    }


def measure(corpus: Path, kind: ExecutorKind, workers: int, positions: bool) -> dict:
    cmd = [sys.executable, __file__, "--single", kind.value, str(workers), str(corpus)]
    if positions:
        cmd.append("--positions")
    done = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if done.returncode != 0:
        sys.exit(f"benchmark run failed:\n{done.stderr}")
    return json.loads(done.stdout.splitlines()[-1])


def cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def machine_info() -> dict:
    gil = getattr(sys, "_is_gil_enabled", lambda: True)()
    return {
        "cpu": cpu_model(),
        "cores": os.cpu_count(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "gil_enabled": gil,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("corpus", type=Path, nargs="?")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--workers", type=int, nargs="+")
    parser.add_argument("--executors", nargs="+", default=["serial", "threads", "processes"])
    parser.add_argument("--positions", action="store_true")
    parser.add_argument("--single", nargs=3, metavar=("EXECUTOR", "WORKERS", "CORPUS"))
    args = parser.parse_args()

    if args.single:
        kind, workers, corpus = args.single
        print(json.dumps(run_once(Path(corpus), ExecutorKind(kind), int(workers), args.positions)))
        return

    if args.corpus is None or args.out is None:
        parser.error("corpus and --out are required")

    worker_counts = args.workers or sorted({1, 2, 4, 8, os.cpu_count() or 1})
    configs = []
    for name in args.executors:
        kind = ExecutorKind(name)
        for workers in [1] if kind is ExecutorKind.serial else worker_counts:
            configs.append((kind, workers))

    print("warm-up run (fills the OS file cache)", file=sys.stderr)
    measure(args.corpus, ExecutorKind.serial, 1, args.positions)

    rows = []
    for kind, workers in configs:
        samples = []
        for n in range(1, args.runs + 1):
            sample = measure(args.corpus, kind, workers, args.positions)
            samples.append(sample)
            print(
                f"{kind.value:<9} x{workers:<3} run {n}/{args.runs}: {sample['wall']:.2f}s",
                file=sys.stderr,
            )
        row = {"executor": kind.value, "workers": workers, "samples": samples}
        for key in ("wall", "cpu", "merge", "plan", "map", "rss_mb"):
            values = [s[key] for s in samples if s[key] is not None]
            row[key] = statistics.median(values) if values else None
        rows.append(row)

    data = {
        "machine": machine_info(),
        "corpus": {
            "path": str(args.corpus),
            "size_mb": corpus_size(args.corpus) / 2**20,
            "docs": rows[0]["samples"][0]["docs"],
            "positions": args.positions,
        },
        "runs": args.runs,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
