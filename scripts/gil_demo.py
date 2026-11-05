"""Lab 5 experiments: the GIL in four timings, a data race, and threads on I/O.

Run it on the normal interpreter and on the free-threaded one and compare:

    uv run python scripts/gil_demo.py
    UV_PROJECT_ENVIRONMENT=.venv-313t uv run --python 3.13t --no-dev python scripts/gil_demo.py
"""

import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import AbstractContextManager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import urlopen


def crunch(_: int = 0) -> int:
    return sum(i * i for i in range(5 * 10**6))


def timed(fn: Callable[[], object]) -> float:
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


def cpu_bound() -> None:
    print("1. CPU-bound work: sum of squares")
    print(f"   one call                         {timed(crunch):.2f}s")
    print(f"   two calls, one after another     {timed(lambda: [crunch(), crunch()]):.2f}s")
    with ThreadPoolExecutor(2) as pool:
        print(
            f"   two threads                      {timed(lambda: list(pool.map(crunch, range(2)))):.2f}s"
        )
    with ProcessPoolExecutor(2) as pool:
        first = timed(lambda: list(pool.map(crunch, range(2))))
        second = timed(lambda: list(pool.map(crunch, range(2))))
    print(f"   two processes, with startup      {first:.2f}s")
    print(f"   two processes, already running   {second:.2f}s")


shared = 0


def add(times: int, lock: AbstractContextManager[bool] | None) -> None:
    global shared
    for _ in range(times):
        if lock:
            with lock:
                shared += 1
        else:
            shared += 1


def count(lock: AbstractContextManager[bool] | None, times: int = 10**6) -> int:
    global shared
    shared = 0
    threads = [threading.Thread(target=add, args=(times, lock)) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return shared


def race() -> None:
    print("\n2. Two threads, 1_000_000 times `shared += 1` each (expected 2000000)")
    print(f"   without a lock                   {count(None)}")
    print(f"   with a lock                      {count(threading.Lock())}")


class SlowHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        time.sleep(0.1)
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, format: str, *args: object) -> None:
        pass


class Server(ThreadingHTTPServer):
    request_queue_size = 64


def io_bound() -> None:
    server = Server(("127.0.0.1", 0), SlowHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}/"

    def fetch(_: int) -> None:
        urlopen(url).read()

    print("\n3. I/O-bound work: 30 requests to a server that answers after 0.1 s")
    for workers in (1, 5, 10, 30):
        with ThreadPoolExecutor(workers) as pool:
            elapsed = timed(lambda: list(pool.map(fetch, range(30))))
        print(f"   {workers:>2} threads                       {elapsed:.2f}s")
    server.shutdown()


def main() -> None:
    gil = getattr(sys, "_is_gil_enabled", lambda: True)()
    print(f"Python {sys.version.split()[0]}, GIL {'on' if gil else 'off'}\n")
    cpu_bound()
    race()
    io_bound()


if __name__ == "__main__":
    main()
