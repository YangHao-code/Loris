"""Exact-order MCC ranking with bitsets and lazy monotone-cover heaps.

For each label/sign, an old heap score is an optimistic bound because novel
coverage can only decrease. Refresh only the head until current; then compare
the two sign heads. Full saturation replaces values with exact zero scores.
"""
import hashlib
import heapq
import numpy as np


def popcount(bits):
    # Campaign's Python3.8 has no int.bit_count. Both branches count exact bits.
    method = getattr(int, 'bit_count', None)
    return method(bits) if method else bin(bits).count('1')


def select_group(items, quota, y, *, diagnostics=None):
    counters = {} if diagnostics is None else diagnostics
    for key in ('popcounts', 'label_heaps_created', 'head_refreshes', 'saturation_rebuilds'):
        counters[key] = 0
    unique, rejected = {}, []
    for item in sorted(items, key=lambda value: value['spec'].key):
        packed = np.packbits(item['mask'], bitorder='little').tobytes()
        key = hashlib.sha256(packed).hexdigest()
        if key in unique:
            rejected.append((item, unique[key]['spec'].key))
        else:
            unique[key] = item
    pool = list(unique.values())
    count = len(pool)
    masks = [int.from_bytes(np.packbits(item['mask'], bitorder='little').tobytes(), 'little') for item in pool]
    supports = [item['support'] for item in pool]
    keys = [item['spec'].key for item in pool]
    used, chosen = set(), []
    covers = [[0, 0] for _ in range(y.shape[1])]
    states = {}

    def novel(mask, cover):
        counters['popcounts'] += 1
        return popcount(mask & ~cover)

    def make_label(label):
        absolute = [abs(float(item['corr'][label])) for item in pool]
        signs = [int(item['corr'][label] >= 0) for item in pool]
        # Stable order reproduces the original tuple's exact secondary rank.
        ordered = sorted(range(count), key=lambda i: (-absolute[i], -supports[i], keys[i]))
        ranks = {index: rank for rank, index in enumerate(ordered)}
        sides = []
        for sign in (0, 1):
            union, heap = 0, []
            for i in range(count):
                if signs[i] != sign:
                    continue
                union |= masks[i]
                if keys[i] in used:
                    continue
                # Same NumPy sqrt and multiplication order as original code.
                value = absolute[i] * np.sqrt(novel(masks[i], covers[label][sign]) / max(1, supports[i]))
                heap.append((-value, ranks[i], i, 0))
            heapq.heapify(heap)
            sides.append({'heap': heap, 'version': 0, 'union': union, 'saturated': False})
        counters['label_heaps_created'] += 1
        states[label] = (absolute, sides)
        return states[label]

    def head(label, sign, absolute, side):
        heap = side['heap']
        cover = covers[label][sign]
        if not side['saturated'] and cover & side['union'] == side['union']:
            # Exact score zero for every remaining mask; preserve secondary rank.
            heap[:] = [(0.0, entry[1], entry[2], side['version']) for entry in heap if keys[entry[2]] not in used]
            heapq.heapify(heap)
            side['saturated'] = True
            counters['saturation_rebuilds'] += 1
        while heap:
            value, rank, i, version = heap[0]
            if keys[i] in used:
                heapq.heappop(heap)
                continue
            if side['saturated'] or version == side['version']:
                return heap[0]
            heapq.heappop(heap)
            score = absolute[i] * np.sqrt(novel(masks[i], cover) / max(1, supports[i]))
            heapq.heappush(heap, (-score, rank, i, side['version']))
            counters['head_refreshes'] += 1
        return None

    diversity_quota = int(np.ceil(quota * 0.75))
    for turn in range(diversity_quota):
        label = turn % y.shape[1]
        absolute, sides = states.get(label) or make_label(label)
        options = []
        for sign in (0, 1):
            candidate = head(label, sign, absolute, sides[sign])
            if candidate is not None:
                options.append((candidate[0], candidate[1], candidate[2], sign))
        if not options:
            break
        _, _, i, sign = min(options)
        heapq.heappop(sides[sign]['heap'])
        used.add(keys[i])
        chosen.append((pool[i], 'label_balanced_complementarity', label, sign))
        old_cover = covers[label][sign]
        covers[label][sign] |= masks[i]
        if old_cover != covers[label][sign]:
            sides[sign]['version'] += 1

    covered = 0
    for item, _, _, _ in chosen:
        # pool references retained without altering any input metadata.
        covered |= int.from_bytes(np.packbits(item['mask'], bitorder='little').tobytes(), 'little')
    # The final coverage stage has one monotone-cover heap and the original
    # fixed secondary order (-max abs corr, -support, stable key).
    remaining = [i for i in range(count) if keys[i] not in used]
    ordered = sorted(remaining, key=lambda i: (-float(np.max(abs(pool[i]['corr']))), -supports[i], keys[i]))
    ranks = {i: rank for rank, i in enumerate(ordered)}
    heap = [(-novel(masks[i], covered), ranks[i], i, 0) for i in remaining]
    heapq.heapify(heap)
    union = 0
    for i in remaining:
        union |= masks[i]
    version, saturated = 0, False
    while len(chosen) < quota and heap:
        if not saturated and covered & union == union:
            heap[:] = [(0, rank, i, version) for _, rank, i, _ in heap if keys[i] not in used]
            heapq.heapify(heap)
            saturated = True
            counters['saturation_rebuilds'] += 1
        value, rank, i, old_version = heapq.heappop(heap)
        if keys[i] in used:
            continue
        if not saturated and old_version != version:
            heapq.heappush(heap, (-novel(masks[i], covered), rank, i, version))
            counters['head_refreshes'] += 1
            continue
        used.add(keys[i])
        chosen.append((pool[i], 'coverage_complementarity', None, None))
        old_cover = covered
        covered |= masks[i]
        if covered != old_cover:
            version += 1
    return chosen, rejected
