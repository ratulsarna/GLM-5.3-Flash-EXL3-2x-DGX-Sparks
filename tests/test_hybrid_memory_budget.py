#!/usr/bin/env python3
"""Exercise the candidate's real padded-cache view on page boundaries."""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import math
import shutil
import tempfile
import unittest
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace


RECIPE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "memory_patch", RECIPE / "overlay/patch_hybrid_memory_budget.py"
)
patch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patch)
SOURCE = None
TENSORS = False
GPU = False
GPU_LOGICAL_BLOCK = patch.DRAFT_BLOCK_SIZE


def reshape_function(torch):
    source = patch.prepare((SOURCE / patch.RESHAPER).read_text(), patch.RESHAPER)
    tree = ast.parse(source)
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_reshape_attention_kv_cache"
    )
    namespace = {
        "torch": torch, "prod": math.prod,
        "get_dtype_size": lambda dtype: torch.empty((), dtype=dtype).element_size(),
    }
    exec("from __future__ import annotations\n" + ast.unparse(function), namespace)
    return namespace[function.name]


def attention_cache_layout(lengths, starts, logical_block):
    # Preserve absolute block indices while recycling physical pages per owner.
    rows = []
    slots = []
    for owner, (length, start) in enumerate(zip(lengths, starts)):
        first = start // logical_block
        stop = math.ceil(length / logical_block)
        pages = [0] * first + [1 + owner + len(lengths) * i
                              for i in range(stop - first)]
        rows.append(pages)
        slots.extend(pages[pos // logical_block] * logical_block + pos % logical_block
                     for pos in range(start, length))
    return rows, slots


class PatchTests(unittest.TestCase):
    def test_both_files_preflight_before_any_write(self):
        with tempfile.TemporaryDirectory(dir=RECIPE) as tmp:
            root = Path(tmp)
            for rel in patch.EDITS:
                (root / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(SOURCE / rel, root / rel)
            planner = root / patch.PLANNER
            before = planner.read_bytes()
            (root / patch.RESHAPER).write_text("unexpected source\n")
            with self.assertRaises(ValueError):
                patch.apply(root)
            self.assertEqual(planner.read_bytes(), before)

    def test_partial_patch_and_duplicate_anchor_fail_closed(self):
        source = (SOURCE / patch.PLANNER).read_text()
        patched = patch.prepare(source, patch.PLANNER)
        self.assertEqual(patch.prepare(patched, patch.PLANNER), patched)
        partial = patched.replace(patch.LAYOUT_NEW, patch.LAYOUT_OLD)
        with self.assertRaises(ValueError):
            patch.prepare(partial, patch.PLANNER)
        with self.assertRaises(ValueError):
            patch.prepare(source + patch.BLOCK_OLD, patch.PLANNER)


class TensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not TENSORS:
            raise unittest.SkipTest("run with --cpu-tensors or --gpu")
        import torch
        cls.torch = torch
        cls.reshape = staticmethod(reshape_function(torch))
        cls.device = "cuda" if GPU else "cpu"

    def test_actual_planner_keeps_pool_bytes_and_admission_bound(self):
        torch = self.torch
        import vllm.v1.core.kv_cache_utils as utils
        from vllm.v1.kv_cache_interface import (
            KpoolTailSpec, MLAAttentionSpec, MambaSpec, SlidingWindowSpec,
        )
        namespace = vars(utils).copy()
        source = patch.prepare((SOURCE / patch.PLANNER).read_text(), patch.PLANNER)
        # Execute the changed owners with the installed spec and sizing classes.
        owners = {"_get_kv_cache_groups_glm5_next", "_glm5_next_tensor_layout",
                  "get_kv_cache_config_from_groups", "_pool_bytes_per_block"}
        for node in ast.parse(source).body:
            if isinstance(node, ast.FunctionDef) and node.name in owners:
                exec("from __future__ import annotations\n" + ast.unparse(node), namespace)
        cfg = SimpleNamespace(
            parallel_config=SimpleNamespace(pipeline_parallel_size=1,
                decode_context_parallel_size=1, prefill_context_parallel_size=1),
            cache_config=SimpleNamespace(num_gpu_blocks_override=None,
                mamba_cache_mode="align", enable_prefix_caching=True,
                prefix_match_unit=None),
            model_config=SimpleNamespace(max_model_len=232000),
            max_in_flight_tokens=4096,
            kv_transfer_config=None,
        )
        specs = {}
        for i in range(11):
            specs[f"mla.{i}"] = MLAAttentionSpec(
                block_size=3584, num_kv_heads=1, head_size=576,
                dtype=torch.uint8, cache_dtype_str="fp8_ds_mla")
            specs[f"index.{i}"] = MLAAttentionSpec(
                block_size=3584, num_kv_heads=1, head_size=132,
                dtype=torch.uint8, compress_ratio=4)
            specs[f"tail.{i}"] = KpoolTailSpec(
                block_size=4, num_kv_heads=1, head_size=128,
                dtype=torch.bfloat16, sliding_window=4)
        for i in range(34):
            specs[f"mamba.{i}"] = MambaSpec(
                block_size=3584, shapes=((1024,),), dtypes=(torch.bfloat16,),
                mamba_cache_mode="align", num_speculative_blocks=7)
        for i in range(5):
            specs[f"draft.{i}"] = SlidingWindowSpec(
                block_size=16, num_kv_heads=4, head_size=128,
                dtype=torch.bfloat16, sliding_window=2048)
        groups = namespace["_get_kv_cache_groups_glm5_next"](cfg, specs)
        per_id = 11 * (2351104 + 118272)
        config = namespace["get_kv_cache_config_from_groups"](
            cfg, groups, 414 * per_id)
        self.assertEqual(config.num_blocks, 414)
        self.assertEqual(sum(t.size for t in config.kv_cache_tensors), 414 * per_id)
        self.assertEqual(namespace["_pool_bytes_per_block"](cfg, groups), per_id)
        counts = [math.ceil(g.kv_cache_spec.max_memory_usage_bytes(cfg)
                            / g.kv_cache_spec.page_size_bytes) for g in groups]
        self.assertEqual(counts, [65, 1, 9, 9, 9, 9,
                                 {512: 13, 896: 8}[patch.DRAFT_BLOCK_SIZE]])
        self.assertEqual(math.lcm(*(g.kv_cache_spec.block_size for g in groups)), 3584)
        self.assertLessEqual(3 * sum(counts), config.num_blocks - 1)
        from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator
        scheduler_config = utils.generate_scheduler_kv_cache_config([config])
        for draft_block in (64, 512, 896):
            variant = deepcopy(scheduler_config)
            variant.kv_cache_groups[-1].kv_cache_spec = replace(
                variant.kv_cache_groups[-1].kv_cache_spec, block_size=draft_block)
            scheduler_unit, hash_unit = utils.resolve_kv_cache_block_sizes(variant, cfg)
            self.assertEqual((scheduler_unit, hash_unit), (3584, draft_block))
            coordinator = HybridKVCacheCoordinator(
                variant, max_model_len=232000, max_in_flight_tokens=4096,
                use_eagle=True, enable_caching=True, enable_kv_cache_events=False,
                dcp_world_size=1, pcp_world_size=1,
                scheduler_block_size=scheduler_unit, hash_block_size=hash_unit)
            # Kpool scratch disables partial hits; the target checkpoint unit stays fixed.
            self.assertFalse(coordinator.enable_partial_hash_hits)
            self.assertEqual(coordinator._cache_hit_alignment_tokens, 3584)

    def view(self, logical=512, kernel=64, order=(0, 2, 1, 3), pages=5):
        torch = self.torch
        page_bytes = 2351104
        raw = torch.full((pages * page_bytes,), 0x5A, dtype=torch.uint8,
                         device=self.device)
        cache_spec = SimpleNamespace(dtype=torch.bfloat16, page_size_bytes=page_bytes,
                                     page_size_padded=page_bytes)
        count = pages * (logical // kernel)
        shape = (count, 4, kernel, 256)
        view = self.reshape(raw, cache_spec, shape, order, count, None)
        return raw, view

    def test_shared_pages_stay_disjoint_and_padding_stays_untouched(self):
        torch = self.torch
        logical = patch.DRAFT_BLOCK_SIZE
        for kernel in (16, 32, 64, logical):
            for order in ((0, 2, 1, 3), (0, 1, 2, 3)):
                with self.subTest(kernel=kernel, order=order):
                    raw, view = self.view(logical=logical, kernel=kernel, order=order)
                    ratio = logical // kernel
                    stride = 2351104 // ratio
                    physical = raw.view(5 * ratio, stride)
                    before = physical.clone()
                    # Nonadjacent pages model three owners and two untouched pages.
                    for owner in (0, 2, 4):
                        view[owner * ratio:(owner + 1) * ratio].fill_(owner + 1)
                    for owner in range(5):
                        span = slice(owner * ratio, (owner + 1) * ratio)
                        if owner in (1, 3):
                            self.assertTrue(torch.equal(physical[span], before[span]))
                        else:
                            self.assertTrue(torch.equal(
                                view[span], torch.full_like(view[span], owner + 1)
                            ))
                            real_bytes = kernel * 2048
                            self.assertTrue(torch.equal(
                                physical[span, real_bytes:], before[span, real_bytes:]
                            ))
                    # Reusing a freed owner's page cannot overwrite another owner.
                    view[2 * ratio:3 * ratio].fill_(19)
                    self.assertTrue(torch.all(view[:ratio] == 1).item())
                    self.assertTrue(torch.all(view[4 * ratio:] == 5).item())

    def test_invalid_split_alignment_and_fit_fail_before_view(self):
        torch = self.torch
        spec = SimpleNamespace(dtype=torch.bfloat16, page_size_bytes=2351104,
                               page_size_padded=2351104)
        raw = torch.empty(2 * spec.page_size_bytes, dtype=torch.uint8,
                          device=self.device)
        for count, shape, message in (
            (3, (3, 4, 64, 256), "tile"),
            (6, (6, 4, 64, 256), "aligned"),
            (16, (16, 4, 512, 256), "exceeds"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                self.reshape(raw, spec, shape, (0, 2, 1, 3), count, None)

    def test_unsplit_existing_padded_page_keeps_its_stride(self):
        raw, view = self.view(logical=64, kernel=64)
        self.assertEqual(view.stride(0) * view.element_size(), 2351104)
        self.assertEqual(view.untyped_storage().data_ptr(), raw.data_ptr())

    def test_cuda_writer_matches_slot_mapping_at_every_boundary(self):
        if not GPU:
            self.skipTest("root runs --gpu")
        torch = self.torch
        from vllm.v1.attention.backends.fa_utils import reshape_and_cache_flash
        logical = GPU_LOGICAL_BLOCK
        for kernel in (64, logical):
            with self.subTest(kernel=kernel):
                raw, view = self.view(logical=logical, kernel=kernel)
                k_cache, v_cache = view.transpose(1, 2).split(128, dim=-1)
                slots = [owner * logical + offset for owner in (0, 2, 4)
                         for offset in (0, 15, 16, 31, 32, 63, 64, 127, 255,
                                        logical - 1)]
                mapping = torch.tensor(slots + [-1], dtype=torch.int64, device="cuda")
                key = torch.arange(len(mapping), dtype=torch.bfloat16, device="cuda")
                key = key[:, None, None].expand(-1, 4, 128).contiguous()
                value = (key + 100).contiguous()
                scale = torch.tensor(1.0, device="cuda")
                before = raw.clone()
                reshape_and_cache_flash(key, value, k_cache, v_cache, mapping,
                                        "auto", scale, scale)
                torch.cuda.synchronize()
                for i, slot in enumerate(slots):
                    page, offset = divmod(slot, kernel)
                    self.assertTrue(torch.equal(k_cache[page, offset], key[i]))
                    self.assertTrue(torch.equal(v_cache[page, offset], value[i]))
                for owner in (1, 3):
                    span = slice(owner * 2351104, (owner + 1) * 2351104)
                    self.assertTrue(torch.equal(raw[span], before[span]))

    def test_cuda_attention_matches_contiguous_and_legacy_cache(self):
        self.assert_cuda_attention_against_legacy((2047, 2048, 2049))

    def test_cuda_short_prompt_and_page_boundaries(self):
        self.assert_cuda_attention_against_legacy((41, 512, 895))
        self.assert_cuda_attention_against_legacy((896, 897, 41))

    def test_cuda_absolute_232k_positions_with_null_prefix(self):
        self.assert_cuda_attention_against_legacy(
            (231998, 231999, 232000), window_only=True)

    def assert_cuda_attention_against_legacy(self, lengths, window_only=False):
        if not GPU:
            self.skipTest("root runs --gpu")
        torch = self.torch
        from vllm.v1.attention.backends.fa_utils import (
            flash_attn_varlen_func, get_flash_attn_version, reshape_and_cache_flash,
        )
        torch.manual_seed(1729)
        logical = GPU_LOGICAL_BLOCK
        scale = torch.tensor(1.0, device="cuda")
        for requested_qlen in (1, 8, 257):
            query_lengths = tuple(min(requested_qlen, length) for length in lengths)
            starts = tuple(max(0, length - qlen - 2047) if window_only else 0
                           for length, qlen in zip(lengths, query_lengths))
            num_stored_tokens = sum(length - start
                                    for length, start in zip(lengths, starts))
            key = torch.randn(num_stored_tokens, 4, 128, dtype=torch.bfloat16,
                              device="cuda")
            value = torch.randn_like(key)
            query = torch.randn(sum(query_lengths), 8, 128, dtype=torch.bfloat16,
                                device="cuda")
            query_offsets = [0]
            for qlen in query_lengths:
                query_offsets.append(query_offsets[-1] + qlen)
            legacy_output = None
            legacy_kv = None
            for manager, kernel in ((64, 64), (logical, 64), (logical, logical)):
                with self.subTest(lengths=lengths, queries=query_lengths,
                                  manager=manager, kernel=kernel):
                    owners, slots = attention_cache_layout(lengths, starts, manager)
                    mapping = torch.tensor(slots, dtype=torch.int64, device="cuda")
                    raw, view = self.view(logical=manager, kernel=kernel,
                                          pages=max(map(max, owners)) + 1)
                    k_cache, v_cache = view.transpose(1, 2).split(128, dim=-1)
                    reshape_and_cache_flash(key, value, k_cache, v_cache, mapping,
                                            "auto", scale, scale)
                    page_indices = torch.div(mapping, kernel, rounding_mode="floor")
                    offsets = mapping.remainder(kernel)
                    stored_key = k_cache[page_indices, offsets].cpu()
                    stored_value = v_cache[page_indices, offsets].cpu()
                    self.assertTrue(torch.equal(stored_key, key.cpu()))
                    self.assertTrue(torch.equal(stored_value, value.cpu()))
                    self.assertTrue(torch.all(raw[:2351104] == 0x5A).item())
                    if manager == 64:
                        legacy_kv = stored_key, stored_value
                    exact_key = torch.equal(stored_key, legacy_kv[0])
                    exact_value = torch.equal(stored_value, legacy_kv[1])
                    print(json.dumps(dict(check="legacy_kv", manager=manager,
                        kernel=kernel, tokens=num_stored_tokens,
                        context_lengths=lengths, query_lengths=query_lengths,
                        stored_from=starts, window_only=window_only,
                        null_prefix_pages=[start // manager for start in starts],
                        key_exact=exact_key, value_exact=exact_value)), flush=True)
                    self.assertTrue(exact_key and exact_value)
                    ratio = manager // kernel
                    rows = [[page * ratio + j for page in pages for j in range(ratio)]
                            for pages in owners]
                    width = max(map(len, rows))
                    table = torch.tensor([r + [0] * (width - len(r)) for r in rows],
                                         dtype=torch.int32, device="cuda")
                    args = dict(q=query, max_seqlen_q=max(query_lengths),
                                cu_seqlens_q=torch.tensor(query_offsets,
                                    dtype=torch.int32, device="cuda"),
                                max_seqlen_k=max(lengths),
                                seqused_k=torch.tensor(lengths, dtype=torch.int32,
                                                       device="cuda"),
                                block_table=table, causal=True,
                                window_size=[2047, 0],
                                fa_version=get_flash_attn_version(head_size=128))
                    actual = flash_attn_varlen_func(k=k_cache, v=v_cache, **args)
                    reference = flash_attn_varlen_func(
                        k=k_cache.contiguous(), v=v_cache.contiguous(), **args)
                    torch.cuda.synchronize()
                    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
                    actual_cpu = actual.cpu()
                    if manager == 64:
                        legacy_output = actual_cpu
                    self.assertEqual(actual_cpu.dtype, torch.bfloat16)
                    delta = actual_cpu.float() - legacy_output.float()
                    rms = delta.square().mean().sqrt().item()
                    baseline_rms = legacy_output.float().square().mean().sqrt().item()
                    relative_rms = rms / baseline_rms
                    # BF16 epsilon bounds relative error; the near-zero floor
                    # covers FP32 accumulation order. The RMS limit is stricter.
                    rtol = torch.finfo(torch.bfloat16).eps
                    atol = 32 * torch.finfo(torch.float32).eps * baseline_rms
                    rms_limit = rtol / 8
                    print(json.dumps(dict(check="legacy_attention", manager=manager,
                        kernel=kernel, query_lengths=query_lengths,
                        context_lengths=lengths, stored_from=starts,
                        window_only=window_only,
                        exact=torch.equal(actual_cpu, legacy_output),
                        max_abs=delta.abs().max().item(), rms=rms,
                        reference_rms=baseline_rms, relative_rms=relative_rms,
                        rtol=rtol, atol=atol, relative_rms_limit=rms_limit)), flush=True)
                    torch.testing.assert_close(actual_cpu, legacy_output,
                                               rtol=rtol, atol=atol)
                    self.assertLessEqual(relative_rms, rms_limit)
                    del raw, view, k_cache, v_cache, stored_key, stored_value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--cpu-tensors", action="store_true")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--logical-block-size", type=int, choices=(512, 896),
                        default=patch.DRAFT_BLOCK_SIZE,
                        help="GPU addressing probe only; does not change the patch")
    args = parser.parse_args()
    SOURCE = args.source_root
    TENSORS = args.cpu_tensors or args.gpu
    GPU = args.gpu
    GPU_LOGICAL_BLOCK = args.logical_block_size
    unittest.main(argv=[__file__], verbosity=2)
