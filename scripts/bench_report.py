"""Turn benchmark JSON files into the README table and the speedup plot.

Speedup is always measured against the serial, 1-worker row of the first file
(the normal GIL build), so rows from the free-threaded build are comparable.

    uv run --group bench python scripts/bench_report.py bench/gil.json bench/nogil.json \
        --png docs/speedup.png
"""

import argparse
import json
from pathlib import Path

ORDER = {"serial": 0, "threads": 1, "processes": 2}


def label(data: dict, executor: str) -> str:
    if data["machine"]["gil_enabled"]:
        return executor
    return f"{executor} (no GIL)"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", type=Path, nargs="+")
    parser.add_argument("--png", type=Path)
    args = parser.parse_args()

    datasets = [json.loads(path.read_text(encoding="utf-8")) for path in args.files]
    baseline = next(
        row["wall"]
        for row in datasets[0]["rows"]
        if row["executor"] == "serial" and row["workers"] == 1
    )

    for data in datasets:
        m, c = data["machine"], data["corpus"]
        print(
            f"- {m['cpu']}, {m['cores']} cores, {m['platform']}, Python {m['python']}, "
            f"GIL {'on' if m['gil_enabled'] else 'off'}"
        )
        print(
            f"  corpus: {c['docs']} documents, {c['size_mb']:.1f} MB, median of {data['runs']} runs"
        )

    print()
    print("| Executor | Workers | Wall (s) | CPU (s) | Peak RSS (MB) | Merge (s) | Speedup |")
    print("|---|---|---|---|---|---|---|")
    for data in datasets:
        for row in sorted(data["rows"], key=lambda r: (ORDER[r["executor"]], r["workers"])):
            rss = f"{row['rss_mb']:.0f}" if row["rss_mb"] is not None else "n/a"
            print(
                f"| {label(data, row['executor'])} | {row['workers']} | {row['wall']:.2f} "
                f"| {row['cpu']:.2f} | {rss} | {row['merge']:.2f} | {baseline / row['wall']:.2f}x |"
            )

    if args.png:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4.5))
        top = 1
        ticks: set[int] = set()
        for data in datasets:
            for executor in ("threads", "processes"):
                rows = sorted(
                    (r for r in data["rows"] if r["executor"] == executor),
                    key=lambda r: r["workers"],
                )
                if rows:
                    xs = [r["workers"] for r in rows]
                    ys = [baseline / r["wall"] for r in rows]
                    top = max(top, max(xs))
                    ticks.update(xs)
                    ax.plot(xs, ys, marker="o", label=label(data, executor))
        ax.plot([1, top], [1, top], "k--", label="ideal")
        ax.set_xticks(sorted(ticks))
        ax.set_xlabel("workers")
        ax.set_ylabel("speedup vs serial")
        ax.set_title("Indexing speedup")
        ax.grid(alpha=0.3)
        ax.legend()
        args.png.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(args.png, dpi=130)
        print(f"\nsaved {args.png}")


if __name__ == "__main__":
    main()
