#!/usr/bin/env python3
"""Host test for patch_mamba_hash_block_split.py (no GPU, no vLLM import).

Applies the overlay to the real scheduler.py text and drives the patched
``_mamba_block_aligned_split`` and ``__init__`` derivations in isolation.
The split cases are the production recipe's (recipe-dedup bf8cbd8 and
0b2810d) plus the chunks measured on the isolated boots.

    GLM53_SCHEDULER_PY=/path/to/vllm/v1/core/sched/scheduler.py \\
        python3 test_mamba_hash_block_split.py

The source may be pristine, carry patch_scheduler_decode_floor.py v5, or
already carry this overlay. ``python3 -m pytest`` runs the same suite; an
absent source is a skip naming the variable.
"""
from __future__ import annotations

import ast
import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
PATCH = next(
    p for p in (HERE / "patch_mamba_hash_block_split.py",
                HERE.parent / "overlay" / "patch_mamba_hash_block_split.py")
    if p.is_file()
)
spec = importlib.util.spec_from_file_location("hash_block_split", PATCH)
patch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patch)
SOURCE = Path(os.environ.get(
    "GLM53_SCHEDULER_PY",
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py"))


def source_text() -> str:
    if not SOURCE.is_file():
        raise unittest.SkipTest(f"set GLM53_SCHEDULER_PY (missing {SOURCE})")
    return SOURCE.read_text()


def unpatched(text: str) -> str:
    """The scheduler as it was before this overlay (identity when pristine)."""
    for _, old, new in reversed(patch.EDITS):
        text = text.replace(new, old, 1)
    return text


class PatchTests(unittest.TestCase):
    def test_apply_is_idempotent_on_the_real_file(self):
        base = unpatched(source_text())
        patched = patch.prepare(base)
        self.assertEqual(patch.prepare(patched), patched)
        self.assertEqual(patched.count(patch.MARK), 2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scheduler.py"
            path.write_text(base)
            env = {**os.environ, "GLM53_SCHEDULER_PY": str(path)}
            for _ in range(2):
                run = subprocess.run([sys.executable, str(PATCH)], env=env,
                                     text=True, capture_output=True)
                self.assertEqual(run.returncode, 0, run.stderr)
                self.assertEqual(path.read_text(), patched)

    def test_drift_partial_state_and_conflicts_fail_closed(self):
        base = unpatched(source_text())
        patched = patch.prepare(base)
        for label, old, new in patch.EDITS:
            with self.subTest(duplicate=label), self.assertRaises(ValueError):
                patch.prepare(base + old)
            with self.subTest(partial=label), self.assertRaises(ValueError):
                patch.prepare(patched.replace(new, old, 1))
            with self.subTest(missing=label), self.assertRaises(ValueError):
                patch.prepare(base.replace(old, "", 1))
        with self.assertRaises(ValueError):
            patch.prepare(base + "\n" + patch.UPSTREAM_CHUNKING_MARK + "\n")
        with self.assertRaises(ValueError):
            patch.prepare(base.replace(patch.DECODE_FLOOR_V5, "# [glm53-decode-floor:v4]")
                          if patch.DECODE_FLOOR_V5 in base
                          else base + "\n# [glm53-decode-floor:v4]\n")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scheduler.py"
            drifted = base + "\n" + patch.UPSTREAM_CHUNKING_MARK + "\n"
            path.write_text(drifted)
            run = subprocess.run([sys.executable, str(PATCH)], text=True, capture_output=True,
                                 env={**os.environ, "GLM53_SCHEDULER_PY": str(path)})
            self.assertNotEqual(run.returncode, 0)
            self.assertEqual(path.read_text(), drifted)


class SchedulerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = patch.prepare(unpatched(source_text()))
        for name, text in (
            ("split", source),
            ("baseline_split", source.replace(patch.STOPS_NEW, patch.STOPS_OLD)
                                     .replace(patch.SPLIT_NEW, patch.SPLIT_OLD)),
        ):
            function = next(node for node in ast.walk(ast.parse(text))
                            if isinstance(node, ast.FunctionDef)
                            and node.name == "_mamba_block_aligned_split")
            namespace = {}
            exec("from __future__ import annotations\n" + ast.unparse(function), namespace)
            setattr(cls, name, staticmethod(namespace[function.name]))
        assignments = [node for node in ast.walk(ast.parse(source))
                       if isinstance(node, ast.Assign)
                       and isinstance(node.targets[0], ast.Attribute)
                       and node.targets[0].attr in ("mamba_block_sizes", "has_mamba_layers")]
        cls.mamba_setup = compile("from __future__ import annotations\n" + ast.unparse(
            ast.Module(body=assignments, type_ignores=[])), patch.SCHEDULER, "exec")

    @staticmethod
    def scheduler(block_size, budget=2048):
        return SimpleNamespace(
            cache_config=SimpleNamespace(block_size=block_size),
            block_size=3584, hash_block_size=64, use_eagle=True,
            mamba_block_sizes=[3584],
            max_num_scheduled_tokens=budget, mamba_partial_cache_hit=False,
            scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
        )

    def chunks(self, length, budget=2048):
        request = SimpleNamespace(num_computed_tokens=0, num_prompt_tokens=length,
                                  num_tokens=length, shared_prefix_boundary=0)
        scheduler = self.scheduler(896, budget)
        chunks = []
        while request.num_computed_tokens < length:
            count = self.split(scheduler, request,
                               min(budget, length - request.num_computed_tokens))
            self.assertGreater(count, 0)
            chunks.append(count)
            request.num_computed_tokens += count
        return chunks

    def test_short_prefills_and_232k_checkpoint_stops(self):
        cases = {
            41: [41], 66: [66], 218: [128, 90], 895: [768, 127],
            896: [832, 64], 897: [832, 65], 1553: [1472, 81],
            232000: [2048, 1536] * 64 + [2048, 512, 64],
        }
        for length, expected in cases.items():
            with self.subTest(length=length):
                self.assertEqual(self.chunks(length), expected)

    def test_measured_production_chunks(self):
        # Head-log chunks of the old production build (578b077e) on the
        # fixed-context screen and the cold long-state medium-revisions prompt;
        # the served budget is 2048 minus 7 draft slots.
        self.assertEqual(self.chunks(1650, 2041), [1536, 114])
        self.assertEqual(self.chunks(12277, 2041),
                         [1984, 1600, 1984, 1600, 1984, 1600, 1408, 117])
        for length in (1650, 12277, 28736, 232000):
            with self.subTest(length=length):
                ends = list(itertools_accumulate(self.chunks(length, 2041)))
                self.assertTrue(all(end % 64 == 0 for end in ends[:-1]))
                self.assertTrue(all(boundary in ends
                                    for boundary in range(3584, ends[-1], 3584)))

    def test_resumed_prefill_shared_junctions_and_decode_keep_baseline_splits(self):
        # Resumes and fair-prefill budgets can start within either page size.
        cases = (
            (232000, 232000, 3584, 0, 0, 2048, 0),
            (232000, 232000, 0, 3584, 0, 2048, 0),
            (232000, 232000, 0, 0, 3584, 2048, 0),
            (232000, 232000, 65, 0, 0, 384, 0),
            (232000, 232000, 896, 0, 0, 256, 0),
            (232000, 232000, 512, 0, 0, 2048, 900),
            (232000, 232000, 3584, 0, 0, 2048, 4500),
            (232000, 232000, 231936, 0, 0, 64, 0),
            (218, 305, 218, 0, 0, 86, 0),
            (218, 305, 304, 0, 0, 8, 0),
        )
        for prompt, tokens, start, local, external, budget, junction in cases:
            with self.subTest(case=(prompt, tokens, start, local, external, budget, junction)):
                request = SimpleNamespace(num_computed_tokens=start,
                    num_prompt_tokens=prompt, num_tokens=tokens,
                    shared_prefix_boundary=junction)
                expected = self.baseline_split(self.scheduler(64, budget), request,
                                               budget, local, external)
                actual = self.split(self.scheduler(896, budget), request,
                                    budget, local, external)
                self.assertEqual(actual, expected)

    def test_actual_mamba_groups_define_boundaries_independently_of_scalar_and_lcm(self):
        class MambaSpec:
            def __init__(self, size):
                self.block_size = size

        scheduler = self.scheduler(896)
        scheduler.cache_config.mamba_block_size = 64
        scheduler.block_size = 10752
        groups = [SimpleNamespace(kv_cache_spec=spec) for spec in
                  (MambaSpec(3584), MambaSpec(768), MambaSpec(3584),
                   SimpleNamespace(block_size=4))]
        exec(self.mamba_setup, dict(self=scheduler, MambaSpec=MambaSpec,
                                  kv_cache_config=SimpleNamespace(kv_cache_groups=groups)))
        self.assertEqual(scheduler.mamba_block_sizes, [768, 3584])
        request = SimpleNamespace(num_computed_tokens=1984, num_prompt_tokens=232000,
                                  num_tokens=232000, shared_prefix_boundary=0)
        self.assertEqual(self.split(scheduler, request, 2041), 320)
        self.assertTrue(scheduler.has_mamba_layers)

    def test_checkpoint_stops_survive_resumes_fair_caps_and_speculative_decode(self):
        cases = (
            # start, local hit, external hit, input budget, fair cap, expected
            (55552, 0, 0, 2041, None, 1792),
            (0, 55552, 0, 2041, None, 1792),
            (0, 0, 55552, 2041, None, 1792),
            (56064, 0, 0, 1536, 1536, 1280),
            (57088, 0, 0, 384, 384, 256),
            (57280, 0, 0, 256, 256, 64),
        )
        for eagle in (False, True):
            for start, local, external, budget, cap, expected in cases:
                with self.subTest(eagle=eagle, start=start, local=local, external=external):
                    scheduler = self.scheduler(896)
                    scheduler.use_eagle = eagle
                    scheduler._glm53_align_prefill_limit = cap
                    request = SimpleNamespace(num_computed_tokens=start,
                        num_prompt_tokens=232000, num_tokens=232000, shared_prefix_boundary=0)
                    count = self.split(scheduler, request, budget, local, external)
                    self.assertEqual(count, expected)
                    self.assertEqual((start + local + external + count) % 4, 0)
        scheduler = self.scheduler(896)
        request = SimpleNamespace(num_computed_tokens=57342, num_prompt_tokens=50000,
                                  num_tokens=57343, shared_prefix_boundary=0)
        self.assertEqual(self.split(scheduler, request, 8), 8)

    def test_caps_below_one_hash_block_still_advance(self):
        # decode-floor v5 hands the per-request cap to the split; a cap below
        # 64 tokens advances sub-block and re-aligns at the next boundary.
        for cap in (1, 8, 32, 63):
            with self.subTest(cap=cap):
                scheduler = self.scheduler(896)
                scheduler._glm53_align_prefill_limit = cap
                request = SimpleNamespace(num_computed_tokens=0, num_prompt_tokens=4000,
                                          num_tokens=4000, shared_prefix_boundary=0)
                ends = []
                while request.num_computed_tokens < 256:
                    count = self.split(scheduler, request, cap)
                    self.assertGreater(count, 0)
                    request.num_computed_tokens += count
                    ends.append(request.num_computed_tokens)
                self.assertIn(64, ends)
                self.assertIn(128, ends)


def itertools_accumulate(values):
    total = 0
    for value in values:
        total += value
        yield total


if __name__ == "__main__":
    if not SOURCE.is_file():
        sys.exit(f"missing {SOURCE}; set GLM53_SCHEDULER_PY")
    unittest.main(argv=[sys.argv[0]], verbosity=2)
