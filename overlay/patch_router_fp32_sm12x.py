#!/usr/bin/env python3
"""Compute the MoE router logits in fp32 on SM12x (GB10), as the checkpoint asks.

GLM-5.3-Flash sets ``moe_router_dtype: "float32"`` and vLLM builds the gate
with ``out_dtype=torch.float32`` (``deepseek_v2._get_moe_router_dtype``; its
own comment: GLM-5 requires fp32 routing). ``GateLinear`` then picks a GEMM
tier, and on SM12x every fp32 tier is gated off: tiers 1-3 need SM90+
specialized kernels and tier 5 (cuBLAS bf16 x bf16 -> fp32,
``torch.mm(..., out_dtype=torch.float32)``) is enabled only through
``allow_specialized_router_gemm`` (SM90/SM100). The gate falls to tier 6, a
bf16 ``F.linear`` whose result is ROUNDED TO BF16 and only then cast to fp32.
Every router logit on this box is therefore a bf16 value (measured:
bf16-exact in 100% of entries, all 45 layers), up to ~1.6e-2 away from the
fp32 product. With sigmoid scoring, top-8 of 288 and 11-24% of tokens per
layer within 1e-3 of the 8th/9th expert boundary, that rounding moves expert
choices the reference implementation would not move.

The checkpoint stores the gate weight in bf16 and the score-correction bias
in fp32, so the reference logits are the fp32 product of bf16 values, which
is exactly what tier 5 computes. On GB10 tier 5 works, is deterministic, and
is faster than tier 6 (no separate cast kernel): 8.8 vs 12.5 us at 8 tokens,
47.9 vs 55.6 us at 1536 tokens; max abs error vs fp64 3.3e-5 vs 1.6e-2.

This patch adds that same product as a tier ahead of tier 6, only when:
the device is SM12x, out_dtype is fp32, weight and input are bf16, and the
gate has no bias (tier 5 ignores bias). Everywhere else nothing changes.
``GLM53_ROUTER_FP32=0`` switches it off at runtime for a rollback without
rebuilding the image (read once per process).

Anchor is this image's ``vllm/model_executor/layers/fused_moe/router/
gate_linear.py``. Idempotent; fails closed on drift.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

P = Path(
    os.environ.get(
        "GLM53_GATE_LINEAR_PY",
        "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/router/gate_linear.py",
    )
)
MARK = "# [glm53-router-fp32-sm12x]"
OLD = """        # Tier 6: F.linear (ReplicatedLinear)
        if self.out_dtype is not None and x.dtype != self.weight.dtype:
"""
NEW = """        # [glm53-router-fp32-sm12x] Tier 5b: tier 5's cuBLAS bf16 x bf16 -> fp32
        # product on SM12x, where tier 5 is gated off and tier 6 would round
        # the router logits to bf16 before the fp32 cast.
        if (
            self.out_dtype == torch.float32
            and self.weight.dtype == torch.bfloat16
            and x.dtype == torch.bfloat16
            and self.bias is None
            and _glm53_sm12x_fp32_router()
        ):
            return torch.mm(x, self.weight.T, out_dtype=torch.float32), None

""" + OLD
TAIL = '''

_GLM53_SM12X_FP32_ROUTER: bool | None = None


def _glm53_sm12x_fp32_router() -> bool:  # [glm53-router-fp32-sm12x]
    """SM12x CUDA device and not switched off by GLM53_ROUTER_FP32=0."""
    global _GLM53_SM12X_FP32_ROUTER
    if _GLM53_SM12X_FP32_ROUTER is None:
        import os

        _GLM53_SM12X_FP32_ROUTER = (
            os.environ.get("GLM53_ROUTER_FP32", "1") != "0"
            and current_platform.is_cuda()
            and current_platform.is_device_capability_family(120)
        )
    return _GLM53_SM12X_FP32_ROUTER
'''
ANCHORS = (
    "from vllm.platforms import current_platform",
    "class GateLinear(ReplicatedLinear):",
    "            output = torch.mm(x, self.weight.T, out_dtype=torch.float32)",
)


def apply_text(src: str) -> tuple[str, str]:
    """Return (new_source, status): applied|skipped|missing:..."""
    if MARK in src:
        if src.count(NEW) != 1 or "def _glm53_sm12x_fp32_router" not in src:
            return src, "missing:partial"
        return src, "skipped"
    if src.count(OLD) != 1:
        return src, "missing:tier6-anchor"
    for anchor in ANCHORS:
        if anchor not in src:
            return src, f"missing:{anchor.strip()[:40]}"
    return src.replace(OLD, NEW, 1) + TAIL, "applied"


def apply_file(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    new, status = apply_text(text)
    if status == "applied":
        compile(new, str(path), "exec")
        path.write_text(new, encoding="utf-8")
    return status


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv if argv is None else argv
    if len(argv) > 1 and argv[1] == "--status":
        target = Path(argv[2]) if len(argv) > 2 else P
        applied = target.is_file() and MARK in target.read_text(encoding="utf-8")
        print("router-fp32-sm12x              :", "APPLIED" if applied else "NOT APPLIED")
        return 0
    target = Path(argv[1]) if len(argv) > 1 else P
    if not target.is_file():
        print(f"[router-fp32-sm12x] missing {target}", file=sys.stderr)
        return 1
    status = apply_file(target)
    print(f"[router-fp32-sm12x] {status}: {target}")
    return 0 if status in ("applied", "skipped") else 1


if __name__ == "__main__":
    sys.exit(main())
