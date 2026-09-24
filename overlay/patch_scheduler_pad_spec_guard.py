#!/usr/bin/env python3
"""Keep a padded spec-decode newcomer consistent with its final chunk size.

The pinned scheduler pads a waiting request with exactly one token left to
compute (a 1-token prompt, or a prefix-cache hit that leaves one token) to
1 + num_spec_tokens tokens while other requests decode, and records
num_spec_tokens -1 placeholders for it in scheduled_spec_decode_tokens. The
point is a uniform decode batch.

Several caps run after that padding and can shrink num_new_tokens again: the
decode-floor v5 fair grant (cap_for -> _target clamps every rung to the
remaining prefill, so it returns 1 here), the long-prefill threshold, the
Mamba block split and encoder budgets. The padding flag survived them, so the
step scheduled 1 token with 7 placeholder drafts. The worker then gave the
request 8 logits ending at its 1-token query (logits_start = query_end - 8),
and the sampler's hidden_states[input_batch.logits_indices] gather
(gpu/model_runner.py sample) read rows before the start of the batch: a
vectorized_gather_kernel index assert and EngineDeadError on both ranks.

This overlay adds one check right before KV allocation: if the chunk no longer
equals 1 + num_spec_tokens, drop the padding and schedule the request as a
plain prefill of its real remaining tokens. When nothing trimmed the chunk the
stock padding is unchanged.

Idempotent (marker comment), fails closed on drifted anchors. Apply after
patch_scheduler_decode_floor.py and patch_mamba_hash_block_split.py.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

SCHED = Path(
    os.environ.get(
        "GLM53_SCHEDULER_PY",
        "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py",
    )
)
MARK = "# [glm53-pad-spec-guard]"

# The stock padding this guard protects; must be present exactly once.
PAD_SET = """                        num_new_tokens = 1 + self.num_spec_tokens
                        if (
                            num_new_tokens > request_token_budget
                            or num_computed_tokens + num_new_tokens > self.max_model_len
                        ):
                            # Prefer to not schedule than schedule un-padded here.
                            break
                        pad_spec_decode = True
"""
PAD_USE = """                if pad_spec_decode:
                    scheduled_spec_decode_tokens[request_id] = [
                        -1
                    ] * self.num_spec_tokens
"""
ANCHOR = """                # During async KV load, no forward pass is run yet.
                # Allocate speculative lookahead slots later to avoid
                # mismatching local and remote block counts.
                limit_lookahead_tokens = load_kv_async and self.num_lookahead_tokens > 0
"""
GUARD = """                # [glm53-pad-spec-guard] Padding schedules this newcomer as a
                # uniform spec decode: 1 real token + num_spec_tokens
                # placeholders. A later cap (mixed-prefill grant, long-prefill
                # threshold, Mamba split, encoder budget) may have trimmed the
                # chunk; the placeholders would then describe tokens that are
                # not in the batch. Schedule the real tokens as a plain prefill.
                if pad_spec_decode and num_new_tokens != 1 + self.num_spec_tokens:
                    pad_spec_decode = False
                    num_new_tokens = min(
                        num_new_tokens, request.num_tokens - num_computed_tokens
                    )
"""


def patch(text: str) -> str:
    for label, needle in (("padding", PAD_SET), ("placeholders", PAD_USE), ("allocation anchor", ANCHOR)):
        n = text.count(needle)
        if n != 1:
            raise SystemExit(f"{SCHED}: expected one {label} block, found {n}")
    return text.replace(ANCHOR, GUARD + ANCHOR, 1)


def main() -> int:
    if not SCHED.is_file():
        raise SystemExit(f"missing {SCHED}")
    text = SCHED.read_text()
    if MARK in text:
        # Marker present: the installed guard must still be intact.
        if text.count(GUARD + ANCHOR) != 1:
            raise SystemExit(f"{SCHED}: {MARK} present but the guard block drifted")
        print(f"{SCHED.name}: {MARK} already present — skipping")
        return 0
    new = patch(text)
    compile(new, str(SCHED), "exec")
    SCHED.write_text(new)
    print(f"patched {SCHED.name} {MARK}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
