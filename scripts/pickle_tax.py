"""Lab 5: what does it cost to move a partial index from a worker to the parent?

Builds one chunk, then pickles it two ways: with the default dataclass pickling
and with Posting.__reduce__ (what findex uses). The parent has to unpickle
every partial one after another, so this load time is the sequential part
that Amdahl's law is about.

    uv run python scripts/pickle_tax.py data/big.jsonl
"""

import argparse
import pickle
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from findex.corpus import plan_chunks
from findex.parallel import build_partial, gc_paused


@dataclass(frozen=True, slots=True)
class PlainPosting:
    doc_id: int
    tf: int
    positions: tuple[int, ...] = ()


def timed[T](fn: Callable[[], T]) -> tuple[float, T]:
    start = time.perf_counter()
    result = fn()
    return time.perf_counter() - start, result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", type=Path)
    args = parser.parse_args()

    chunk = plan_chunks(args.corpus, 1)[0]
    with gc_paused():
        build, partial = timed(lambda: build_partial(chunk))
    count = sum(len(plist) for plist in partial.postings.values())
    plain = {
        term: [PlainPosting(p.doc_id, p.tf, p.positions) for p in plist]
        for term, plist in partial.postings.items()
    }
    print(f"build_partial: {build:.2f}s, {count:,} postings\n")

    print("| pickled as | dumps (s) | loads (s) | size (MB) |")
    print("|---|---|---|---|")
    load = 0.0
    for name, data in (("default dataclass", plain), ("Posting.__reduce__", partial.postings)):
        with gc_paused():
            dump, blob = timed(lambda data=data: pickle.dumps(data, pickle.HIGHEST_PROTOCOL))
            load, _ = timed(lambda blob=blob: pickle.loads(blob))
        print(f"| {name} | {dump:.2f} | {load:.2f} | {len(blob) / 2**20:.1f} |")

    # the same load, but with the cycle collector running
    with_gc, _ = timed(lambda: pickle.loads(blob))
    print("\n(both measured with the garbage collector paused, as findex runs;")
    print(f" loading the __reduce__ blob with it on takes {with_gc:.2f}s instead of {load:.2f}s)")
    print(
        f"\nthe parent unpickles every partial sequentially: {load:.2f}s of {build:.2f}s serial work"
    )
    print(f"so Amdahl caps the speedup at about {build / load:.1f}x, however many workers")


if __name__ == "__main__":
    main()
