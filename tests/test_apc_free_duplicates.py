#!/usr/bin/env python3
"""CPU cache ownership and repeated-prefix eviction checks; no GPU execution."""

import argparse
import ast
from bisect import bisect_right
import hashlib
import inspect
import itertools
import json
from pathlib import Path
import runpy
import sys
import textwrap
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).parent))
import test_apc_periodic_retention as owners

overlay = runpy.run_path(str(Path(__file__).parents[1] / "overlay/patch_apc_free_duplicates.py"))


class ChurnCache(owners.DraftCache):
    pass


# Reuse the owner harness's allocation/free calls with recorded prefill ends.
for method in (owners.DraftCache.step, owners.DraftCache.run):
    tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
    if method.__name__ == "step":
        end = tree.body[0].body[0]
        assert isinstance(end, ast.Assign) and ast.unparse(end.targets[0]) == "end"
        end.value = ast.parse("req.chunk_ends[bisect_right(req.chunk_ends, processed)]", mode="eval").body
    else:
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and ast.unparse(node) == "req.num_prompt_tokens":
                node.attr = "total_tokens"
    ast.fix_missing_locations(tree)
    scope = dict(vars(owners), bisect_right=bisect_right)
    exec(compile(tree, "<churn-driver>", "exec"), scope)
    setattr(ChurnCache, method.__name__, scope[method.__name__])


def clone(name, canonical, prompt, total):
    req = owners.request(name, total, canonical, total)
    req.num_prompt_tokens = prompt
    req.total_tokens = total
    req.chunk_ends = []
    position = 0
    tail = prompt // 64 * 64 - 64
    while position < prompt:
        end = min(prompt, position + 1984, (position // 3584 + 1) * 3584)
        if position < tail < end:
            end = tail
        req.chunk_ends.append(end)
        position = end
    while position < total:
        position = min(total, position + 8, (position // 3584 + 1) * 3584)
        req.chunk_ends.append(position)
    return req


def churn(ns):
    cache = ChurnCache(ns, 57344, 418, 896)
    originals = [owners.request(f"original-{i}", 229000) for i in range(3)]
    cold_ends = list(itertools.accumulate([1984, 1600] * 63 + [1984, 1152, 72]))
    for i, req in enumerate(originals):
        cold = clone(f"cold-{i}", req, 229000, 229000)
        assert cold.chunk_ends == cold_ends
        cache.run([cold])
    previous = originals
    for generation, size in enumerate((229256, 229512, 229768)):
        current = [owners.request(f"growth-{generation}-{i}", size, req,
                                  req.num_prompt_tokens) for i, req in enumerate(previous)]
        for i, req in enumerate(current):
            cache.run([clone(f"grow-{generation}-{i}", req, size, size)])
        previous = current
    count = [owners.request(f"count-{i}", 231063, req, 228977)
             for i, req in enumerate(originals)]
    snapshots = []
    for cycle in range(1, 13):
        cache.run([clone(f"decode-{cycle}-{i}", req, 229015, 231063)
                   for i, req in enumerate(count)])
        cache.run([clone(f"return-{cycle}-{i}", req, 229000, 229000)
                   for i, req in enumerate(originals)])
        hits = [cache.hit(owners.request(f"edit-{i}", 229000, req, 206078))[1]
                for i, req in enumerate(originals)]
        duplicates = sum(len(v) - 1 for v in cache.pool.cached_block_hash_to_block._cache.values()
                         if isinstance(v, dict))
        snapshots.append({"cycle": cycle, "grid_missing": cache.missing_grid(originals),
                          "late_edit_hits": hits, "extra_copies": duplicates})
    return {"peak_live": cache.peak_live, "snapshots": snapshots}


class DuplicateTests(unittest.TestCase):
    ns = None
    source = None

    def setUp(self):
        self.pool = self.ns["BlockPool"](8, True, 64)

    def key(self, name, group=2):
        return self.ns["make_block_hash_with_group_id"](hashlib.sha256(name.encode()).digest(), group)

    def insert(self, block, name, group=2):
        self.pool._insert_block_hash(self.key(name, group), block, 3584)

    def queue(self):
        queue = self.pool.free_block_queue
        cursor = queue.fake_free_list_head.next_free_block
        ids = []
        while cursor is not queue.fake_free_list_tail:
            self.assertNotIn(cursor.block_id, ids)
            self.assertEqual(cursor.ref_cnt, 0)
            self.assertIs(cursor.next_free_block.prev_free_block, cursor)
            ids.append(cursor.block_id)
            cursor = cursor.next_free_block
        self.assertEqual(len(ids), self.pool.get_num_free_blocks())
        return ids

    def test_release_preserves_request_and_copy_pins(self):
        first, second = self.pool.get_new_blocks(2)
        self.insert(first, "shared")
        self.insert(second, "shared")
        self.pool.touch([first])
        self.pool.free_blocks([second])
        self.assertIsNotNone(second.block_hash)
        self.pool.free_blocks([first])
        self.assertEqual(first.ref_cnt, 1)
        self.assertIsNotNone(first.block_hash)
        survivor_order = self.queue()
        self.pool.free_blocks([first])
        self.assertIsNone(first.block_hash)
        self.assertEqual(self.queue(), [first.block_id] + survivor_order)
        self.assertIs(self.pool.get_new_blocks(1)[0], first)
        self.assertIs(self.pool.cached_block_hash_to_block.get_one_block(self.key("shared")), second)

    def test_same_batch_keeps_first_released_copy(self):
        blocks = self.pool.get_new_blocks(3)
        for block in blocks:
            self.insert(block, "shared")
        self.pool.free_blocks(blocks)
        self.assertIsNotNone(blocks[0].block_hash)
        self.assertTrue(all(b.block_hash is None for b in blocks[1:]))
        self.assertEqual(self.queue()[:2], [b.block_id for b in blocks[1:]])
        self.assertIs(self.pool.cached_block_hash_to_block.get_one_block(self.key("shared")), blocks[0])
        # A one-entry dictionary remains after removing the other copies.
        self.assertFalse(self.pool.cached_block_hash_to_block.has_other_free_block(
            self.key("shared"), blocks[0].block_id))

    def test_unique_alias_keeps_the_block(self):
        first, second = self.pool.get_new_blocks(2)
        self.insert(first, "shared")
        self.insert(second, "shared")
        self.insert(second, "unique-alias")
        self.pool.free_blocks([first])
        self.pool.free_blocks([second])
        self.assertIsNotNone(first.block_hash)
        self.assertIsNotNone(second.block_hash)
        self.assertIs(self.pool.cached_block_hash_to_block.get_one_block(self.key("unique-alias")), second)
        self.queue()

    def test_all_aliases_survive_without_removed_events(self):
        first, second, third = self.pool.get_new_blocks(3)
        self.insert(first, "shared")
        self.insert(second, "shared")
        self.insert(second, "alias")
        self.insert(third, "alias")
        self.pool.free_blocks([first, third])
        before = self.queue()
        events = []
        evictions = []
        self.pool._emit_block_removed_events = events.extend
        self.pool.metrics_collector = SimpleNamespace(
            on_block_evicted=lambda b: evictions.append(b.block_id),
            on_block_allocated=lambda b: None)
        self.pool.free_blocks([second])
        self.assertEqual(events, [])
        self.assertEqual(evictions, [])
        self.assertNotIn(second.block_id, self.pool.cached_block_hashes_by_block)
        self.assertEqual(self.queue(), [second.block_id] + before)
        self.assertIs(self.pool.get_new_blocks(1)[0], second)
        self.assertEqual(evictions, [second.block_id])
        self.assertIs(self.pool.cached_block_hash_to_block.get_one_block(self.key("shared")), first)
        self.assertIs(self.pool.cached_block_hash_to_block.get_one_block(self.key("alias")), third)

    def test_groups_and_low_priority_order_are_preserved(self):
        first, second, third = self.pool.get_new_blocks(3)
        self.insert(first, "same", 0)
        self.insert(second, "same", 2)
        self.insert(third, "draft", 6)
        self.pool.low_priority_cache_group_ids = frozenset({6})
        self.pool.free_blocks([first, second, third])
        self.assertTrue(all(b.block_hash is not None for b in (first, second, third)))
        order = self.queue()
        self.assertEqual(order[0], third.block_id)
        self.assertEqual(order[-2:], [first.block_id, second.block_id])

    def test_patch_is_idempotent_and_rejects_partial_or_drifted_source(self):
        prepare = overlay["prepare"]
        self.assertEqual(prepare(self.source), self.source)
        original = self.source
        for old, new in overlay["EDITS"]:
            original = original.replace(new, old)
        self.assertEqual(prepare(original), self.source)
        for old, new in overlay["EDITS"]:
            with self.assertRaises(ValueError):
                prepare(self.source.replace(new, old))
        with self.assertRaises(ValueError):
            prepare(original.replace(overlay["MAP_OLD"], "    def drifted(self):\n"))

    def test_twelve_repeated_decode_cycles(self):
        candidate = churn(self.ns)
        for row in candidate["snapshots"]:
            self.assertEqual(row["grid_missing"], [[], [], []], row)
            self.assertEqual(row["late_edit_hits"], [172032] * 3, row)
            self.assertEqual(row["extra_copies"], 0, row)
        original = self.source
        for old, new in overlay["EDITS"]:
            original = original.replace(new, old)
        control_ns = dict(self.ns)
        nodes = [n for n in ast.parse(original).body
                 if getattr(n, "name", None) in ("BlockPool", "BlockHashToBlockMap")]
        exec("from __future__ import annotations\n" + ast.unparse(ast.Module(body=nodes, type_ignores=[])), control_ns)
        control = churn(control_ns)
        self.assertEqual(control["snapshots"][3]["late_edit_hits"], [0, 0, 0])
        print("CHURN", json.dumps({"usable_ids": 417, "candidate": candidate, "control": control}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    DuplicateTests.ns = owners.load_owners(args.source_dir)
    DuplicateTests.source = (args.source_dir / "block_pool.py").read_text()
    unittest.main(argv=[__file__, *remaining])
