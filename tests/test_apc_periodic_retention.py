#!/usr/bin/env python3
"""CPU cache-owner checks against extracted image source; no model import.

Usage: python3 test_apc_periodic_retention.py --source-dir /path/to/core-files
The source directory must contain the four named vLLM files loaded below.
This exercises metadata and eviction, not GPU execution or numeric quality.
"""

from __future__ import annotations

import argparse
import ast
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import hashlib
import itertools
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple, overload


PAGE = 3584
PROMPT = 232000


class AttentionSpec:
    block_size = PAGE


class FullAttentionSpec(AttentionSpec):
    pass


class ChunkedLocalAttentionSpec(FullAttentionSpec):
    pass


class MambaSpec:
    block_size = PAGE
    num_speculative_blocks = 7
    mamba_cache_mode = "align"


class SlidingWindowSpec(AttentionSpec):
    block_size = 64
    sliding_window = 2048


class KpoolTailSpec(SlidingWindowSpec):
    block_size = 4


def load_owners(source_dir):
    ns = dict(globals(), BlockHash=bytes, BlockHashWithGroupId=bytes)
    ns.update(cdiv=lambda x, y: -(-x // y), logger=logging.getLogger("cache-test"))
    selections = {
        "kv_cache_utils.py": {
            "KVCacheBlock", "FreeKVCacheBlockQueue", "BlockHashListWithBlockSize",
            "make_block_hash_with_group_id", "get_block_hash", "get_group_id",
            "resolve_block_hashes",
        },
        "block_pool.py": {"BlockHashToBlockMap", "BlockPool"},
        "single_type_kv_cache_manager.py": {
            "SingleTypeKVCacheManager", "FullAttentionManager", "MambaManager",
            "SlidingWindowManager",
            "KpoolTailManager",
        },
        "kv_cache_coordinator.py": {
            "SpecGroup", "_glm53_inner_kv_spec", "_glm53_is_draft_swa_spec",
            "_glm53_dflash_replay_safe_hit",
        },
    }
    for filename, names in selections.items():
        text = (source_dir / filename).read_text()
        tree = ast.parse(text)
        nodes = [node for node in tree.body if getattr(node, "name", None) in names]
        assert {node.name for node in nodes} == names, filename
        if filename == "kv_cache_coordinator.py":
            cls = next(n for n in tree.body if getattr(n, "name", None) == "HybridKVCacheCoordinator")
            method = next(n for n in cls.body if getattr(n, "name", None) == "find_longest_cache_hit")
            nodes.append(method)
            cache_method = next(n for n in cls.body if getattr(n, "name", None) == "cache_blocks")
            cache_method.name = "hybrid_cache_blocks"
            nodes.append(cache_method)
        module = ast.Module(body=nodes, type_ignores=[])
        exec("from __future__ import annotations\n" + ast.unparse(module), ns)
        print(f"SOURCE {filename} sha256={hashlib.sha256(text.encode()).hexdigest()}")
    return ns


def request(name, tokens=PROMPT, original=None, common_tokens=0):
    hashes = []
    previous = b""
    for i in range(-(-tokens // 64)):
        if original is not None and (i + 1) * 64 <= common_tokens:
            previous = original.block_hashes[i]
        else:
            previous = hashlib.sha256(previous + f"{name}:{i}".encode()).digest()
        hashes.append(previous)
    return SimpleNamespace(request_id=name, num_prompt_tokens=tokens,
                           shared_prefix_boundary=0, block_hashes=hashes)


class Cache:
    def __init__(self, ns, interval, physical_blocks=414):
        self.ns = ns
        self.interval = interval
        self.pool = ns["BlockPool"](physical_blocks, True, 64)
        self.pool.low_priority_cache_group_ids = frozenset({6})
        self.targets = [ns["FullAttentionManager"](FullAttentionSpec(), self.pool, True, 0, PAGE)]
        for gid in range(2, 6):
            manager = ns["MambaManager"](MambaSpec(), self.pool, enable_caching=True,
                                          kv_cache_group_id=gid, scheduler_block_size=PAGE)
            manager._glm53_retain_previous_dflash_boundary = interval != PAGE
            self.targets.append(manager)
        specs = [FullAttentionSpec(), MambaSpec(), SlidingWindowSpec()]
        group = ns["SpecGroup"]
        self.coordinator = SimpleNamespace(
            kv_cache_config=SimpleNamespace(kv_cache_groups=[None] * 7),
            attention_groups=[group(specs[0], [0], ns["FullAttentionManager"], False),
                              group(specs[1], [2, 3, 4, 5], ns["MambaManager"], False),
                              group(specs[2], [6], ns["SlidingWindowManager"], True)],
            single_type_managers=[self.targets[0], None, *self.targets[1:],
                                  SimpleNamespace(block_size=64)],
            block_pool=self.pool, _cache_hit_alignment_tokens=PAGE,
            hash_block_size=64, enable_partial_hash_hits=False,
            dcp_world_size=1, dflash_swa_replay_tokens=2048,
            dflash_boundary_group_ids=frozenset(),
        )

    def hit(self, req):
        return self.ns["find_longest_cache_hit"](self.coordinator, req.block_hashes,
                                                  req.num_prompt_tokens - 1)

    def step(self, req, processed):
        end = min(req.num_prompt_tokens, processed + 2048,
                  (processed // PAGE + 1) * PAGE)
        for manager in self.targets:
            manager.new_step_starts()
            manager.remove_skipped_blocks(req.request_id, processed, req.num_prompt_tokens)
            manager.allocate_new_blocks(req.request_id, end, end)
            manager.cache_blocks(req, end // PAGE * PAGE, self.interval)
        return end

    def fill(self, req, scratch=0):
        # Fixed uncacheable scratch is an explicit pressure input, not a model
        # of the drafter allocator. Target managers execute their real paths.
        extra = self.pool.get_new_blocks(scratch)
        processed = 0
        while processed < req.num_prompt_tokens:
            processed = self.step(req, processed)
        for manager in self.targets:
            manager.free(req.request_id)
        self.pool.free_blocks(extra)

    def fill_concurrent(self, requests, scratch):
        extra = self.pool.get_new_blocks(scratch * len(requests))
        processed = [0] * len(requests)
        while any(n < req.num_prompt_tokens for n, req in zip(processed, requests)):
            for i, req in enumerate(requests):
                if processed[i] < req.num_prompt_tokens:
                    processed[i] = self.step(req, processed[i])
        for req in requests:
            for manager in self.targets:
                manager.free(req.request_id)
        self.pool.free_blocks(extra)


class DraftCache(Cache):
    """Real seven-group metadata paths with bounded delayed draft release."""

    def __init__(self, ns, interval, physical_blocks=414, draft_block_size=512):
        super().__init__(ns, interval, physical_blocks)
        draft_spec = SlidingWindowSpec()
        draft_spec.block_size = draft_block_size
        draft = ns["SlidingWindowManager"](draft_spec, block_pool=self.pool,
                                           enable_caching=True, kv_cache_group_id=6,
                                           scheduler_block_size=PAGE)
        draft.use_eagle = True
        kpool = ns["KpoolTailManager"](KpoolTailSpec(), self.pool, True, 1, PAGE)
        self.managers = [self.targets[0], kpool, *self.targets[1:], draft]
        self.free_order = [6, 0, 1, 2, 3, 4, 5]
        self.coordinator.single_type_managers = self.managers
        self.coordinator.attention_groups[-1] = ns["SpecGroup"](
            draft_spec, [6], ns["SlidingWindowManager"], True)
        self.coordinator.enable_partial_hash_hits = False
        self.coordinator.scheduler_block_size = PAGE
        self.coordinator.retention_interval_by_group = (interval,) * 6 + (0,)
        self.peak_live = 0

    def start(self, req):
        blocks, hit, _ = self.hit(req)
        for manager, group_blocks in zip(self.managers, blocks):
            manager.add_local_computed_blocks(req.request_id, group_blocks, hit, 0)
        return hit

    def step(self, req, processed):
        end = min(req.num_prompt_tokens, processed + 2048,
                  (processed // PAGE + 1) * PAGE)
        for manager in self.managers:
            manager.new_step_starts()
        for gid in self.free_order:
            # Keep one extra chunk of draft pages live, matching the two-chunk
            # reservation. This does not emulate asynchronous GPU execution.
            committed = max(0, processed - 2048) if gid == 6 else processed
            self.managers[gid].remove_skipped_blocks(req.request_id, committed,
                                                    req.num_prompt_tokens)
        for manager in self.managers:
            manager.allocate_new_blocks(req.request_id, end, end)
        self.ns["hybrid_cache_blocks"](self.coordinator, req, end)
        self.peak_live = max(self.peak_live, sum(b.ref_cnt > 0 and not b.is_null
                                               for b in self.pool.blocks))
        return end

    def finish(self, req):
        for gid in self.free_order:
            self.managers[gid].free(req.request_id)

    def run(self, requests, cancel=None):
        starts = [self.start(req) for req in requests]
        processed = list(starts)
        live = set(range(len(requests)))
        steps = 0
        while live:
            for i in sorted(live):
                req = requests[i]
                if cancel == (i, steps):
                    self.finish(req)
                    live.remove(i)
                    continue
                if processed[i] < req.num_prompt_tokens:
                    processed[i] = self.step(req, processed[i])
                if processed[i] == req.num_prompt_tokens:
                    self.finish(req)
                    live.remove(i)
            steps += 1
        assert self.pool.get_num_free_blocks() == self.pool.num_gpu_blocks - 1
        assert all(b.ref_cnt == 0 for b in self.pool.blocks if not b.is_null)
        assert all(not m.req_to_blocks for m in self.managers)
        return starts

    def missing_grid(self, requests):
        result = []
        for req in requests:
            hashes = self.ns["resolve_block_hashes"](req.block_hashes, 64, PAGE)
            result.append([n for n in range(self.interval, req.num_prompt_tokens, self.interval)
                           if self.pool.get_cached_block(hashes[n // PAGE - 1], [2, 3, 4, 5]) is None])
        return result


def check_branches(ns):
    cache = Cache(ns, 28672, 2048)
    original = request("original")
    cache.fill(original)
    assert cache.hit(original)[1] == 64 * PAGE
    hits = {}
    for common in [0, 1, 28671, 28672, 28673, 116000, 208800, 229376]:
        branch = request(f"edit-{common}", original=original, common_tokens=common)
        hit = cache.hit(branch)[1]
        assert hit <= common, (common, hit)
        assert hit >= common // 28672 * 28672, (common, hit)
        hits[common] = hit
    # A missing group invalidates the full checkpoint, even if the other three
    # states and all attention pages are still cached.
    full_hashes = ns["resolve_block_hashes"](original.block_hashes, 64, PAGE)
    missing = cache.pool.get_cached_block(full_hashes[55], [4])[0]
    cache.pool.evict_blocks({missing.block_id})
    edited = request("missing-group", original=original, common_tokens=208800)
    assert cache.hit(edited)[1] == 48 * PAGE
    # A branch can share live state with the original; freeing one owner must
    # leave the other's references protected from allocation pressure.
    blocks, hit, _ = cache.hit(original)
    shared = [block for group in blocks for block in group if not block.is_null]
    cache.pool.touch(shared)
    cache.pool.touch(shared)
    cache.pool.free_blocks(shared)
    pressure = cache.pool.get_new_blocks(cache.pool.get_num_free_blocks())
    assert cache.hit(original)[1] == hit
    assert all(block.ref_cnt == 1 for block in shared)
    cache.pool.free_blocks(pressure)
    cache.pool.free_blocks(shared)
    print(f"PASS edits, missing Mamba group, and concurrent shared refs: {hits}")

    cache = Cache(ns, 28672, 2048)
    cache.fill(original)
    branch = request("branch", original=original, common_tokens=208800)
    cache.fill(branch)
    assert cache.hit(original)[1] == 64 * PAGE
    assert cache.hit(branch)[1] == 64 * PAGE
    while first_mla := cache.pool.get_cached_block(full_hashes[0], [0]):
        cache.pool.evict_blocks({first_mla[0].block_id})
    assert cache.hit(original)[1] == 0
    print("PASS original branch survives sibling fill; missing first MLA page forces zero")


def check_replay(ns):
    cache = Cache(ns, 28672, 2048)
    short = request("short", 64 * PAGE + 65)
    cache.fill(short)
    assert cache.hit(short)[1] == 63 * PAGE
    assert short.num_prompt_tokens - cache.hit(short)[1] == PAGE + 65
    # A longer request may reuse only the part already computed and cached.
    partial = request("partial", 40 * PAGE + 10)
    cache.fill(partial)
    longer = request("resumed", original=partial, common_tokens=partial.num_prompt_tokens)
    assert cache.hit(longer)[1] == 40 * PAGE
    print("PASS missing draft window backs up one page; partial history resumes safely")


def capacity_sweep(ns):
    for interval in [0, 14336, 28672, 43008, 50176, 57344]:
        for scratch, concurrent in [(98, False), (14, False), (14, True)]:
            cache = Cache(ns, interval)
            histories = [request(f"history-{i}") for i in range(3)]
            if concurrent:
                cache.fill_concurrent(histories, scratch=scratch)
            else:
                for history in histories:
                    cache.fill(history, scratch=scratch)
            hits = [cache.hit(history)[1] for history in histories]
            edit_hits = []
            missing_periodic = []
            for history in histories:
                edit_hits.append(cache.hit(request("edit", original=history,
                                                    common_tokens=208800))[1])
                if interval:
                    hashes = ns["resolve_block_hashes"](history.block_hashes, 64, PAGE)
                    missing_periodic.append([i * PAGE for i in range(interval // PAGE, 65, interval // PAGE)
                                             if cache.pool.get_cached_block(hashes[i - 1], [2, 3, 4, 5]) is None])
            assert all(0 <= hit <= 64 * PAGE for hit in hits)
            if interval == 57344 and scratch == 14:
                assert hits == [64 * PAGE] * 3
                assert missing_periodic == [[], [], []]
                assert edit_hits == [48 * PAGE] * 3
            print(f"CAPACITY interval={interval} scratch={scratch} concurrent={concurrent} "
                  f"hits={hits} edit90_hits={edit_hits} missing_periodic={missing_periodic}")


def check_draft512_lifecycle(ns):
    for usable, draft_size, interval in [(413, 512, 28672), (413, 512, 50176),
                                         (413, 512, 57344), (402, 512, 50176),
                                         (402, 512, 57344), (402, 512, 114688),
                                         (402, 896, 57344)]:
        cache = DraftCache(ns, interval, usable + 1, draft_size)
        originals = [request(f"live-{i}") for i in range(3)]
        assert cache.run(originals) == [0, 0, 0]
        missing = cache.missing_grid(originals)
        returned = [request(f"return-{i}", original=req, common_tokens=PROMPT)
                    for i, req in enumerate(originals)]
        starts = cache.run(returned)
        assert starts == [64 * PAGE] * 3
        after_return = cache.missing_grid(originals)
        # Cancel a cache return after one chunk while the other two complete.
        returns2 = [request(f"cancel-return-{i}", original=req, common_tokens=PROMPT)
                    for i, req in enumerate(originals)]
        assert cache.run(returns2, cancel=(0, 1)) == [64 * PAGE] * 3
        after_cancel = cache.missing_grid(originals)
        branch = request("edited-live", original=originals[0], common_tokens=208800)
        branch_start = cache.run([branch])[0]
        original_hits = [cache.hit(req)[1] for req in originals]
        after_branch = cache.missing_grid(originals)
        returns3 = [request(f"post-branch-return-{i}", original=req, common_tokens=PROMPT)
                    for i, req in enumerate(originals)]
        assert cache.run(returns3) == [64 * PAGE] * 3
        after_branch_return = cache.missing_grid(originals)
        print(f"LIFECYCLE draft={draft_size} usable={usable} interval={interval} peak_live={cache.peak_live} "
              f"returns={starts} branch_start={branch_start} originals={original_hits} "
              f"missing_cold={missing} missing_return={after_return} "
              f"missing_cancel={after_cancel} missing_branch={after_branch} "
              f"missing_post_branch_return={after_branch_return}")
        if interval == 57344:
            assert missing == after_return == after_cancel == after_branch == [[], [], []]
            assert original_hits == [64 * PAGE] * 3
            assert branch_start == 48 * PAGE
            assert cache.peak_live <= 345
        if draft_size == 896 or interval == 114688:
            assert after_branch_return == [[], [], []]
    print("PASS real draft512/Kpool allocation, three returns, cancellation, and sibling preservation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    args = parser.parse_args()
    ns = load_owners(args.source_dir)
    check_branches(ns)
    check_replay(ns)
    capacity_sweep(ns)
    check_draft512_lifecycle(ns)
    print("PASS CPU cache-owner checks; GPU quality and capacity remain untested")


if __name__ == "__main__":
    main()
