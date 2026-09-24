#!/usr/bin/env python3
"""CPU regression test for patch_scheduler_pad_spec_guard.py.

Runs the pinned scheduler's real waiting-request admission code (from
`pad_spec_decode = False` through the guard), extracted with ast from the
installed source, against the real decode-floor v5 fair policy class from the
same source. Only KV allocation and the rest of the loop are left out.

The failing production step: request A decodes, request B arrives with one
token left to compute. Stock padding makes B 1 + 7 tokens with 7 placeholder
drafts, then the fair grant clamps B to its 1 remaining token. Without the
guard B leaves admission with 1 token and pad_spec_decode still set.

Source: GLM53_SCHEDULER_PY_SRC, else the installed vLLM scheduler.
"""
from __future__ import annotations

import ast
import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PATCH = next(
    (p for p in (HERE / "patch_scheduler_pad_spec_guard.py",
                 ROOT / "overlay" / "patch_scheduler_pad_spec_guard.py") if p.is_file()),
    None,
)
if PATCH is None:
    raise SystemExit("missing patch_scheduler_pad_spec_guard.py")
MARK = "# [glm53-pad-spec-guard]"
NUM_SPEC = 7
FAIR_ENV = {
    "GLM53_MIXED_PREFILL_CHUNK": "fair",
    "GLM53_FAIR_PREFILL_CHUNK": "256",
    "GLM53_FAIR_PREFILL_SHARE": "0.30",
    "GLM53_FAIR_PREFILL_MAX_INTERVAL_MS": "2000",
    "GLM53_FAIR_PREFILL_MAX_STEP_MS": "2000",
    "GLM53_FAIR_PREFILL_MAX_CHUNKS": "1",
}


def source_path() -> Path:
    for p in (Path(os.environ.get("GLM53_SCHEDULER_PY_SRC", "/missing")),
              Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py")):
        if p.is_file():
            return p
    raise SystemExit("Set GLM53_SCHEDULER_PY_SRC to the pinned scheduler source")


def unguarded_and_guarded() -> tuple[str, str]:
    text = source_path().read_text()
    if MARK in text:
        start = text.index("                " + MARK)
        end = text.index("                # During async KV load, no forward pass is run yet.", start)
        clean = text[:start] + text[end:]
    else:
        clean = text
    with tempfile.TemporaryDirectory() as temp:
        target = Path(temp) / "scheduler.py"
        target.write_text(clean)
        env = {**os.environ, "GLM53_SCHEDULER_PY": str(target)}
        subprocess.run([sys.executable, str(PATCH)], env=env, check=True, capture_output=True)
        guarded = target.read_text()
        # Idempotent.
        subprocess.run([sys.executable, str(PATCH)], env=env, check=True, capture_output=True)
        assert target.read_text() == guarded
        # A marker with a drifted guard fails closed and leaves the file alone.
        drifted = guarded.replace("pad_spec_decode = False\n                    num_new_tokens = min(",
                                  "pad_spec_decode = False\n                    num_new_tokens = max(", 1)
        target.write_text(drifted)
        r = subprocess.run([sys.executable, str(PATCH)], env=env, capture_output=True)
        assert r.returncode != 0 and target.read_text() == drifted
        # Drifted stock padding fails closed.
        target.write_text(clean.replace("pad_spec_decode = True\n", "pad_spec_decode = 1\n", 1))
        r = subprocess.run([sys.executable, str(PATCH)], env=env, capture_output=True)
        assert r.returncode != 0
    return clean, guarded


def admission(source: str):
    """Compile the admission statements of the waiting loop as one-shot code."""
    tree = ast.parse(source)
    schedule = next(n for n in ast.walk(tree)
                    if isinstance(n, ast.FunctionDef) and n.name == "schedule")
    for node in ast.walk(schedule):
        if not isinstance(node, (ast.While, ast.For)):
            continue
        body = node.body
        starts = [i for i, s in enumerate(body) if ast.unparse(s) == "pad_spec_decode = False"]
        if not starts:
            continue
        ends = [i for i, s in enumerate(body)
                if ast.unparse(s).startswith("limit_lookahead_tokens = ")]
        segment = body[starts[0]:ends[0]]
        done = ast.parse("_reached_allocation = True").body
        once = ast.For(target=ast.Name("_once", ast.Store()),
                       iter=ast.Tuple([ast.Constant(0)], ast.Load()),
                       body=segment + done, orelse=[])
        module = ast.fix_missing_locations(ast.Module(body=[once], type_ignores=[]))
        return compile(module, "<pinned-waiting-admission>", "exec")
    raise AssertionError("waiting admission segment not found")


def fair_policy_class(source: str):
    begin = source.index("class _Glm53MixedPrefill:")
    end = source.index("_GLM53_MIXED = _Glm53MixedPrefill()")
    ns = {"os": os, "time": __import__("time")}
    exec(source[begin:end], ns)
    return ns["_Glm53MixedPrefill"]


class Clock:
    t = 1000.0

    def __call__(self):
        return self.t


class Req:
    def __init__(self, rid, prompt, computed=0, decode=False):
        self.request_id = rid
        self.num_prompt_tokens = prompt
        self.num_computed_tokens = computed
        self.num_tokens = prompt + int(decode)
        self.spec_token_ids = [-1] * NUM_SPEC if decode else []
        self.num_output_placeholders = 0
        self.next_decode_eligible_step = 0
        self.max_tokens = 4000
        self.has_encoder_inputs = False
        self.is_prefill_chunk = not decode

    @property
    def num_tokens_with_spec(self):
        return self.num_tokens + len(self.spec_token_ids)


class Queue:
    def __init__(self, items):
        self.items = list(items)

    def pop_request(self):
        return self.items.pop(0)

    def prepend_request(self, r):
        self.items.insert(0, r)


class Sched:
    def __init__(self, running, waiting):
        self.running = list(running)
        self.waiting = list(waiting)
        self.skipped_waiting = []
        self.requests = {r.request_id: r for r in self.running + self.waiting}
        self.current_step = 1
        self.max_model_len = 232000
        self.num_spec_tokens = NUM_SPEC
        self.dynamic_sd_lookup = None
        self.num_sampled_tokens_per_step = 1
        self.need_mamba_block_aligned_split = False
        self.scheduler_config = SimpleNamespace(
            long_prefill_token_threshold=0, enable_chunked_prefill=True)


class GuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clean, cls.guarded = unguarded_and_guarded()
        cls.policy_cls = fair_policy_class(cls.guarded)

    def run_admission(self, source, prompt, computed=0, cap=None, real_policy=False):
        a = Req("A", 92, 92, decode=True)
        b = Req("B", prompt, computed)
        s = Sched([a], [b])
        if real_policy:
            with patch.dict(os.environ, FAIR_ENV), contextlib.redirect_stdout(io.StringIO()):
                policy = self.policy_cls(now=Clock())
            policy.hist_every = 0
            policy.begin_step(s)
            cap_fn = lambda sched, r, c=None: policy.cap_for(sched, r, c)  # noqa: E731
        else:
            policy = SimpleNamespace(
                mode="fair", needs_prefill_compute=lambda r, c=None: r.num_tokens - (c or 0) > 0,
                note_scheduled=lambda r, n: None)
            cap_fn = lambda sched, r, c=None: cap  # noqa: E731
        ns = dict(
            self=s, request=b, request_queue=Queue([b]), step_skipped_waiting=Queue([]),
            _GLM53_MIXED=policy, _glm53_mixed_prefill_policy=cap_fn,
            num_computed_tokens=computed, num_new_local_computed_tokens=0,
            num_external_computed_tokens=0, load_kv_async=False, defer_prefills=False,
            token_budget=2048 - 3, input_budget=2048 - 3, draft_slots=8,
            scheduled_running_reqs=[a], prefill_scheduled=False,
            encoder_compute_budget=0,
        )
        exec(admission(source), ns)
        return ns

    def test_fair_grant_trims_padded_newcomer_stock_bug(self):
        # The pinned source without the guard reproduces the production step.
        ns = self.run_admission(self.clean, prompt=1, real_policy=True)
        self.assertTrue(ns.get("_reached_allocation"))
        self.assertEqual(ns["num_new_tokens"], 1)
        self.assertTrue(ns["pad_spec_decode"])

    def test_guard_schedules_trimmed_newcomer_as_plain_prefill(self):
        ns = self.run_admission(self.guarded, prompt=1, real_policy=True)
        self.assertTrue(ns.get("_reached_allocation"))
        self.assertEqual(ns["num_new_tokens"], 1)
        self.assertFalse(ns["pad_spec_decode"])

    def test_guard_covers_prefix_hit_leaving_one_token(self):
        # 4,097-token prompt with a 4,096-token prefix hit: one token left.
        for source, padded in ((self.clean, True), (self.guarded, False)):
            ns = self.run_admission(source, prompt=4097, computed=4096, real_policy=True)
            self.assertEqual(ns["num_new_tokens"], 1)
            self.assertIs(ns["pad_spec_decode"], padded)

    def test_untrimmed_padding_is_unchanged(self):
        ns = self.run_admission(self.guarded, prompt=1, cap=None)
        self.assertEqual(ns["num_new_tokens"], 1 + NUM_SPEC)
        self.assertTrue(ns["pad_spec_decode"])

    def test_any_trim_below_padded_size_drops_padding(self):
        for cap in (1, 2, 5, NUM_SPEC):
            ns = self.run_admission(self.guarded, prompt=1, cap=cap)
            self.assertEqual(ns["num_new_tokens"], 1, cap)
            self.assertFalse(ns["pad_spec_decode"], cap)

    def test_deferred_newcomer_is_not_admitted(self):
        ns = self.run_admission(self.guarded, prompt=1, cap=0)
        self.assertNotIn("_reached_allocation", ns)

    def test_multi_token_prefill_is_untouched(self):
        ns = self.run_admission(self.guarded, prompt=300, cap=256)
        self.assertEqual(ns["num_new_tokens"], 256)
        self.assertFalse(ns["pad_spec_decode"])


def main() -> int:
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(GuardTests))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
