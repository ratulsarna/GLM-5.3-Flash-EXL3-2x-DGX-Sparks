#!/usr/bin/env python3
"""Split prompts at the hash block and stop at every Mamba checkpoint.

``Scheduler._mamba_block_aligned_split`` keeps the "align" cache-mode
invariant (slot p holds the SSM state after exactly (p + 1) * block_size
tokens, written at chunk ends) using ``cache_config.block_size``. The engine
recomputes that value as the smallest prefix-caching block, which the compact
DFlash drafter group drags down to its own page (896). Two edits restore the
split the production build has served since recipe-dedup:

* Chunk ends align to ``hash_block_size`` (the 64-token prefix-match unit),
  not the drafter page. Every chunk but a prompt's last is then a whole
  number of 64-token blocks, so prompt splitting does not change with the
  drafter page size.
* Every Mamba block boundary (3584 here) is a mandatory stop. The manager
  hashes a Mamba block as soon as the computed prefix covers it, and align
  mode writes state only at chunk ends. Without the stop, a chunk that
  crosses a boundary registers that boundary's block holding a state from
  another position, so a prefix hit resumes the SSM layers at the wrong
  token count.

``last_cache_position`` keeps the stock EAGLE back-off. At the 64-token hash
block it only moves one chunk end by 64 tokens; every Mamba checkpoint is
already a stop, so no reusable state depends on it.

Requires patch_scheduler_decode_floor.py v5 (or an unpatched scheduler): v5
bounds the alignment's ``max_prefill_tokens`` by the per-request mixed cap,
so a request capped below one hash block keeps sub-block progress instead of
rounding to zero. Earlier decode-floor forms are rejected. It must not be
combined with patch_mamba_align_chunking.py, which aligns chunks to the
Mamba block instead.

Usage:
    python3 patch_mamba_hash_block_split.py

Idempotent; a partially applied or drifted source fails before writing.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

P = Path(
    os.environ.get(
        "GLM53_SCHEDULER_PY",
        "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py",
    )
)
SCHEDULER = "v1/core/sched/scheduler.py"
MARK = "# [glm53-mamba-hash-block-split-v1]"
DECODE_FLOOR_MARK = "# [glm53-decode-floor"  # every decode-floor version
DECODE_FLOOR_V5 = "# [glm53-decode-floor:v5]"
UPSTREAM_CHUNKING_MARK = "# [glm53-mamba-align-chunking-v1]"

MAMBA_IMPORT_OLD = "from vllm.v1.kv_cache_interface import KVCacheConfig\n"
MAMBA_IMPORT_NEW = "from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec\n"

MAMBA_SIZES_OLD = """        self.has_mamba_layers = kv_cache_config.has_mamba_layers
"""
MAMBA_SIZES_NEW = """        # [glm53-mamba-hash-block-split-v1] The Mamba groups' own block sizes:
        # each of their boundaries is a checkpoint the split must stop at.
        self.mamba_block_sizes = sorted({
            group.kv_cache_spec.block_size
            for group in kv_cache_config.kv_cache_groups
            if isinstance(group.kv_cache_spec, MambaSpec)
        })
        self.has_mamba_layers = bool(self.mamba_block_sizes)
"""

SPLIT_OLD = """        block_size = self.cache_config.block_size
        # The last block-aligned position whose state can be cached. With
"""
SPLIT_NEW = """        block_size = self.hash_block_size  # [glm53-mamba-hash-block-split-v1]
        # The last block-aligned position whose state can be cached. With
"""

STOPS_OLD = """        stops = (
            # Same invariant: a chunk starting mid-block stops at the boundary
"""
STOPS_NEW = """        stops = (
            # KDA stores only chunk-final state; materialize every Mamba boundary.
            min((start // size + 1) * size for size in self.mamba_block_sizes),
            # Same invariant: a chunk starting mid-block stops at the boundary
"""

EDITS = (
    ("import", MAMBA_IMPORT_OLD, MAMBA_IMPORT_NEW),
    ("mamba sizes", MAMBA_SIZES_OLD, MAMBA_SIZES_NEW),
    ("split", SPLIT_OLD, SPLIT_NEW),
    ("stops", STOPS_OLD, STOPS_NEW),
)


def prepare(source: str, relative_path: str = SCHEDULER) -> str:
    """Return the patched scheduler text; raise ValueError on any drift."""
    if relative_path != SCHEDULER:
        raise ValueError(f"{relative_path}: this overlay edits only {SCHEDULER}")
    if UPSTREAM_CHUNKING_MARK in source:
        raise ValueError(
            f"{relative_path}: patch_mamba_align_chunking.py is applied; "
            "it conflicts with the hash-block split"
        )
    if DECODE_FLOOR_MARK in source and DECODE_FLOOR_V5 not in source:
        raise ValueError(
            f"{relative_path}: patch_scheduler_decode_floor.py older than v5 present; "
            "its per-request cap is not applied to the split"
        )
    original = all(source.count(old) == 1 and new not in source for _, old, new in EDITS)
    patched = all(source.count(new) == 1 for _, _, new in EDITS)
    if patched:
        # An applied block may extend its anchor; judge the superseded forms
        # with every applied block removed.
        stripped = source
        for _, _, new in EDITS:
            stripped = stripped.replace(new, "")
        if any(old in stripped for _, old, _ in EDITS):
            raise ValueError(f"{relative_path}: superseded form still present")
        compile(source, relative_path, "exec")
        return source
    if not original:
        raise ValueError(f"{relative_path}: patch anchors drifted or are incomplete")
    for _, old, new in EDITS:
        source = source.replace(old, new, 1)
    compile(source, relative_path, "exec")
    return source


def main() -> int:
    if not P.is_file():
        raise SystemExit(f"missing {P}")
    text = P.read_text()
    try:
        patched = prepare(text)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    if patched != text:
        P.write_text(patched)
    print(f"patched {P.name} ({MARK})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
