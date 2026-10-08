"""Author batch-query extension; budgets count documents, not provider fallbacks."""
import math

CADENCES = ("after_saturation", "each_closure")
QUERY_POOLS = ("uncovered", "all_unanswered", "needs_review")


def remaining_budget(limit, used):
    return None if limit is None else max(0, limit - used)


def batch_size(loop, initial_pool_size, eligible, used):
    fixed = loop.get("query_batch_size")
    requested = fixed if fixed is not None else max(
        loop.get("query_batch_min", 32),
        int(math.ceil(initial_pool_size * loop.get("query_batch_fraction", 0.01))),
    )
    remaining = remaining_budget(loop.get("label_budget"), used)
    return min(requested, eligible) if remaining is None else min(requested, eligible, remaining)


def parse_budget(value):
    if str(value).lower() in ("unlimited", "none", "inf", "infinity"):
        return "unlimited"
    value = int(value)
    if value < 0:
        raise ValueError("budget must be nonnegative or unlimited")
    return value
