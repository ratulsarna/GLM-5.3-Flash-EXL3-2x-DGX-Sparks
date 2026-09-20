#!/usr/bin/env python3
"""Pack 896 draft tokens into each existing GLM shared cache page."""
from __future__ import annotations

import argparse
import os
import stat
from pathlib import Path


PLANNER = "v1/core/kv_cache_utils.py"
RESHAPER = "v1/worker/gpu/attn_utils.py"
SCHEDULER = "v1/core/sched/scheduler.py"
MARK = "[glm53-hybrid-memory-budget]"
DRAFT_BLOCK_SIZE = 896

BLOCK_OLD = """            # PADDED SLOT-SHARE: 656 vs 4096 cannot exact-fill on this MLA
            # block. Manager 64 matches the SWA kernel, so padding the page
            # to mla_page is a safe strided view (boot 8 OOB was kernel 64
            # inside a 2304-token manager). Layer i co-owns MLA tensor i.
            compact_block = 64
"""
BLOCK_NEW = f"""            # [glm53-hybrid-memory-budget] Keep the scheduler alignment while
            # using more of each shared page for the draft attention window.
            compact_block = {DRAFT_BLOCK_SIZE}
            if (
                mla_block % compact_block
                or compact_block * draft_bytes_per_token > mla_page
                or len(draft_specs) > len(mla_names)
            ):
                raise ValueError("GLM draft block{DRAFT_BLOCK_SIZE} does not fit the shared layout")
"""
LAYOUT_OLD = """                s.block_size != 64 or s.page_size_padded != mla_page
"""
LAYOUT_NEW = f"""                # [glm53-hybrid-memory-budget] Kernel pages may split a
                # logical block; the reshape owner divides their stride.
                s.block_size not in (64, {DRAFT_BLOCK_SIZE})
                or s.page_size_padded != mla_page
                or inner[mla_names[0]].block_size % s.block_size
                or s.unpadded_page_size_bytes > mla_page
"""
STRIDE_OLD = """        dtype_size = get_dtype_size(kv_cache_spec.dtype)
        page_stride = kv_cache_spec.page_size_bytes // dtype_size
"""
STRIDE_NEW = """        # [glm53-hybrid-memory-budget] num_blocks counts kernel pages;
        # the allocation and padded spec count logical manager pages.
        dtype_size = get_dtype_size(kv_cache_spec.dtype)
        page_bytes = kv_cache_spec.page_size_bytes
        raw_bytes = kv_raw_tensor.numel() * kv_raw_tensor.element_size()
        if raw_bytes % page_bytes:
            raise ValueError("Padded KV allocation is not whole logical pages")
        logical_pages = raw_bytes // page_bytes
        if logical_pages <= 0 or num_blocks <= 0 or num_blocks % logical_pages:
            raise ValueError("Padded KV kernel pages do not tile logical pages")
        split = num_blocks // logical_pages
        if page_bytes % (split * dtype_size):
            raise ValueError("Padded KV kernel-page stride is not dtype aligned")
        kernel_page_bytes = prod(kv_cache_shape[1:]) * dtype_size
        if kernel_page_bytes > page_bytes // split:
            raise ValueError("Padded KV kernel page exceeds its logical-page share")
        page_stride = page_bytes // split // dtype_size
"""

SPLIT_OLD = """        block_size = self.cache_config.block_size
        # The last block-aligned position whose state can be cached. With
"""
SPLIT_NEW = """        block_size = self.hash_block_size
        # The last block-aligned position whose state can be cached. With
"""

EDITS = {
    PLANNER: ((BLOCK_OLD, BLOCK_NEW), (LAYOUT_OLD, LAYOUT_NEW)),
    RESHAPER: ((STRIDE_OLD, STRIDE_NEW),),
    SCHEDULER: ((SPLIT_OLD, SPLIT_NEW),),
}


def prepare(source: str, relative_path: str) -> str:
    edits = EDITS[relative_path]
    original = all(source.count(old) == 1 and new not in source for old, new in edits)
    patched = all(source.count(new) == 1 and old not in source for old, new in edits)
    if patched:
        compile(source, relative_path, "exec")
        return source
    if not original:
        raise ValueError(f"{relative_path}: patch anchors drifted or are incomplete")
    for old, new in edits:
        source = source.replace(old, new)
    compile(source, relative_path, "exec")
    return source


def apply(root: Path, preflight: bool = False) -> None:
    prepared = []
    for rel in EDITS:
        path = root / rel
        old = path.read_text()
        prepared.append((path, old, prepare(old, rel)))
    if preflight:
        print(f"{MARK} preflight OK")
        return
    for path, old, new in prepared:
        if old == new:
            continue
        temporary = path.with_name(path.name + ".glm53-memory.tmp")
        try:
            temporary.write_text(new)
            temporary.chmod(stat.S_IMODE(path.stat().st_mode))
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        for pyc in (path.parent / "__pycache__").glob(path.stem + ".*.pyc"):
            pyc.unlink()
    print(f"{MARK} applied and source compiled")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root", type=Path,
        default=Path("/usr/local/lib/python3.12/dist-packages/vllm"),
    )
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    apply(args.source_root, args.preflight)
