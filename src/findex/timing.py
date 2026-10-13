import functools
import logging
import time

log = logging.getLogger("findex")


def timed(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start
            log.debug("%s took %.2f ms", fn.__name__, elapsed * 1000)

    return wrapper
