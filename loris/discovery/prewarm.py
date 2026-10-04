"""Warm the original text cache and verify supplied fit masks on the full axis."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import pickle
import sqlite3
import time

import numpy as np
from loris._engine.primitive_masks import TextPrimitiveCache
from loris._engine.rlearner import atomic_json
from loris._engine.rlearner_config import PatternSpec, ParameterDomain
from loris.predicates import predicate_from_dict
from .manifest import bound, file_record, read_json, validate_context

_CONTEXT = None


def _worker(arguments):
    cache_dir, anchors, memory_bytes, disk_bytes = arguments
    cache = TextPrimitiveCache(
        _CONTEXT["docs"],
        cache_dir,
        attribute_presence=_CONTEXT["attribute_presence"],
        memory_bytes=memory_bytes,
        disk_bytes=disk_bytes,
    )
    try:
        for attr, regex in anchors:
            cache.regex_stats(attr, regex)
        return cache.audit()
    finally:
        cache.close()


def prewarm(*, context_receipt, supply, output, workers=4, memory_mib=128, disk_mib=2048):
    global _CONTEXT
    if any(type(v) is not int or v < 1 for v in (workers, memory_mib, disk_mib)):
        raise ValueError("positive worker and memory/disk cache limits required")
    start = time.monotonic()
    receipt_path = Path(context_receipt).resolve(strict=True)
    receipt = read_json(receipt_path)
    if receipt.get("status") != "complete":
        raise ValueError("complete context receipt required")
    context_path = bound(receipt["context"], receipt_path.parent)
    with context_path.open("rb") as stream:
        context = pickle.load(stream)
    validate_context(context)
    supply = Path(supply).resolve(strict=True)
    completion = read_json(supply / "COMPLETE.json")
    if completion.get("status") != "complete":
        raise ValueError("complete fit-only supply required")
    pattern_path = bound(completion["patterns"], supply)
    raw = read_json(pattern_path)
    patterns = {
        cluster: [
            PatternSpec(
                v["key"],
                predicate_from_dict(v["predicate"]),
                ParameterDomain(**v["frequency_domain"]) if v["frequency_domain"] else None,
                tuple(v["frequency_ops"]),
            )
            for v in items
        ]
        for cluster, items in raw.items()
    }
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    cache_dir = out / "lexical_cache"
    memory_bytes, disk_bytes = memory_mib << 20, disk_mib << 20
    _CONTEXT = context
    cache = None
    try:
        flat = {spec.key: spec for items in patterns.values() for spec in items}
        anchors = {}
        for spec in flat.values():
            p = spec.predicate
            for name in ("r", "r1", "r2"):
                regex = getattr(p, name, None)
                if regex is not None:
                    anchors[(p.attr, regex.raw, int(regex.flags))] = (p.attr, regex)
        items = [anchors[key] for key in sorted(anchors)]
        cache = TextPrimitiveCache(
            context["docs"],
            cache_dir,
            attribute_presence=context["attribute_presence"],
            memory_bytes=memory_bytes,
            disk_bytes=disk_bytes,
        )
        cache_id, cache_binding = cache.context_id, cache.binding
        cache.close()
        cache = None
        atomic_json(
            out / "PROTOCOL.json",
            dict(
                schema="loris_original_lexical_cache_prewarm_v1",
                source=file_record(__file__),
                context=file_record(context_path),
                supply=file_record(supply / "COMPLETE.json"),
                workers=workers,
                memory_mib_per_worker=memory_mib,
                disk_mib=disk_mib,
                unique_regexes=len(items),
                unique_pattern_prototypes=len(flat),
                original_binding=cache_binding,
                no_key_translation=True,
                dev_text_for_mask_interpretation_only=True,
                test_access=False,
            ),
        )
        work = [(cache_dir, items[i::workers], memory_bytes, disk_bytes) for i in range(workers)]
        if workers == 1:
            audits = [_worker(work[0])]
        else:
            with ProcessPoolExecutor(
                max_workers=workers, mp_context=multiprocessing.get_context("fork")
            ) as pool:
                audits = list(pool.map(_worker, work))
        cache = TextPrimitiveCache(
            context["docs"],
            cache_dir,
            attribute_presence=context["attribute_presence"],
            memory_bytes=memory_bytes,
            disk_bytes=disk_bytes,
        )
        fit, checked = np.asarray(context["fit_indices"]), 0
        for cluster, specs in patterns.items():
            with np.load(
                supply / ("cluster_%02d" % int(cluster)) / "FIT_MASKS.npz", allow_pickle=False
            ) as saved:
                if not np.array_equal(saved["fit_indices"], fit):
                    raise ValueError("supply fit row axis differs")
                masks = saved["masks"]
                if len(masks) != len(specs):
                    raise ValueError("supply prototype/mask counts differ")
                for spec, expected in zip(specs, masks):
                    if not np.array_equal(cache.mask(spec.predicate)[fit], expected):
                        raise ValueError(
                            "full runtime mask differs from supplied fit mask: " + spec.key
                        )
                    checked += 1
        stats = cache.audit()
        cache.close()
        cache = None
        database = cache_dir / cache_id / "lexical.sqlite"
        with sqlite3.connect(str(database)) as connection:
            counts = dict(
                connection.execute("SELECT namespace,count(*) FROM arrays GROUP BY namespace")
            )
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        result = dict(
            status="complete",
            lexical_cache_dir=str(cache_dir),
            context_id=cache_id,
            workers=audits,
            original_cache_stats=stats,
            entries=counts,
            selected_fit_masks_independently_equal_to_original_full_cache=checked,
            regexes=len(items),
            seconds=time.monotonic() - start,
            source=file_record(__file__),
            protocol=file_record(out / "PROTOCOL.json"),
            cache_database=file_record(database),
            BO_calls=0,
            Chase_calls=0,
            training_calls=0,
        )
        atomic_json(out / "COMPLETE.json", result)
        return result
    except BaseException as error:
        atomic_json(
            out / "FAILURE.json",
            dict(status="failed", type=type(error).__name__, reason=str(error)),
        )
        raise
    finally:
        if cache is not None:
            cache.close()
        _CONTEXT = None


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--context-receipt", required=True)
    cli.add_argument("--supply", required=True)
    cli.add_argument("--output", required=True)
    cli.add_argument("--workers", type=int, default=4)
    cli.add_argument("--memory-mib", type=int, default=128)
    cli.add_argument("--disk-mib", type=int, default=2048)
    result = prewarm(**vars(cli.parse_args(argv)))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
