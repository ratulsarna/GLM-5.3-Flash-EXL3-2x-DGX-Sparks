#!/usr/bin/env python3
"""CPU test: the SM12x fp32-router patcher matches the image's gate_linear.py
anchors, is idempotent, fails closed on drift, and the patched forward returns
the fp32 product (not bf16-rounded logits) only for SM12x + fp32 out_dtype +
bf16 weight/input + no bias, honouring GLM53_ROUTER_FP32=0.

GLM53_REQUIRE_TARGET=1 also preflights the real file named by
GLM53_GATE_LINEAR_PY (default: this image's gate_linear.py) before the patch.
GLM53_TEST_DEVICE=cuda runs the numerics on the GPU (CPU torch builds lack
torch.mm(out_dtype=...), so the image build skips them).
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _d in (HERE, ROOT / "overlay"):
    if (_d / "patch_router_fp32_sm12x.py").is_file():
        sys.path.insert(0, str(_d))
        break
from patch_router_fp32_sm12x import MARK, NEW, OLD, P, apply_text  # noqa: E402

# Dependency-free harness carrying the exact image anchors: tier 5 as in the
# image, tier 6 = bf16 F.linear rounded to bf16, then cast to out_dtype.
MINIMAL = (
    "import torch\n"
    "from vllm.platforms import current_platform\n"
    "\n"
    "class ReplicatedLinear:\n"
    "    def forward(self, x):\n"
    "        return torch.nn.functional.linear(x, self.weight), None\n"
    "\n"
    "class GateLinear(ReplicatedLinear):\n"
    "    def __init__(self, weight, out_dtype, bias=None):\n"
    "        self.weight = weight\n"
    "        self.out_dtype = out_dtype\n"
    "        self.bias = bias\n"
    "        self.allow_cublas_router_gemm = False\n"
    "\n"
    "    def forward(self, x):\n"
    "        # Tier 5: cuBLAS bf16→fp32\n"
    "        if self.allow_cublas_router_gemm and x.dtype == torch.bfloat16:\n"
    "            output = torch.mm(x, self.weight.T, out_dtype=torch.float32)\n"
    "            return output, None\n"
    "\n"
    f"{OLD}"
    "            x = x.to(self.weight.dtype)\n"
    "        output, output_bias = super().forward(x)\n"
    "        if self.out_dtype is not None and output.dtype != self.out_dtype:\n"
    "            output = output.to(self.out_dtype)\n"
    "        return output, output_bias\n"
)


class _Platform:
    def __init__(self, family: int) -> None:
        self.family = family

    def is_cuda(self) -> bool:
        return True

    def is_device_capability_family(self, capability: int) -> bool:
        return capability // 10 == self.family // 10


def _load(family: int) -> dict:
    import types

    out, status = apply_text(MINIMAL)
    assert status == "applied", status
    platforms = types.ModuleType("vllm.platforms")
    platforms.current_platform = _Platform(family)
    sys.modules.setdefault("vllm", types.ModuleType("vllm"))
    sys.modules["vllm.platforms"] = platforms
    ns: dict = {}
    exec(compile(out, "patched_gate_linear_fixture.py", "exec"), ns)
    return ns


def test_apply_then_skip() -> None:
    out, status = apply_text(MINIMAL)
    assert status == "applied", status
    assert out.count(MARK) == 2, out.count(MARK)
    assert out.count(NEW) == 1
    out2, status2 = apply_text(out)
    assert status2 == "skipped", status2
    assert out2 == out


def test_missing_anchor_fails_closed() -> None:
    _, status = apply_text(MINIMAL.replace("# Tier 6: F.linear", "# Tier 7: F.linear"))
    assert status.startswith("missing:"), status
    _, status = apply_text(MINIMAL.replace("from vllm.platforms import current_platform\n", ""))
    assert status.startswith("missing:"), status
    partial = MINIMAL.replace(OLD, OLD + "        pass  " + MARK + "\n")
    _, status = apply_text(partial)
    assert status == "missing:partial", status


def _run_case(family: int, out_dtype, wdtype, xdtype, bias=None, env=None):
    import torch

    saved = os.environ.get("GLM53_ROUTER_FP32")
    if env is None:
        os.environ.pop("GLM53_ROUTER_FP32", None)
    else:
        os.environ["GLM53_ROUTER_FP32"] = env
    try:
        ns = _load(family)
        torch.manual_seed(0)
        device = os.environ.get("GLM53_TEST_DEVICE", "cpu")
        w = (torch.randn(288, 256) * 0.05).to(device=device, dtype=wdtype)
        x = torch.randn(16, 256).to(device=device, dtype=xdtype)
        gate = ns["GateLinear"](w, out_dtype, bias)
        y, _ = gate.forward(x)
        ref = torch.nn.functional.linear(x.double(), w.double())
        return y, ref
    finally:
        if saved is None:
            os.environ.pop("GLM53_ROUTER_FP32", None)
        else:
            os.environ["GLM53_ROUTER_FP32"] = saved


def _cpu_mm_out_dtype_supported() -> bool:
    import torch

    device = os.environ.get("GLM53_TEST_DEVICE", "cpu")
    try:
        one = torch.ones(2, 2, dtype=torch.bfloat16, device=device)
        torch.mm(one, one, out_dtype=torch.float32)
        return True
    except (RuntimeError, TypeError, NotImplementedError):
        return False


def test_numerics() -> bool:
    import torch

    if not _cpu_mm_out_dtype_supported():
        print(f"numerics: SKIP (torch.mm out_dtype unsupported on {os.environ.get('GLM53_TEST_DEVICE', 'cpu')})")
        return False
    bf16, fp32 = torch.bfloat16, torch.float32
    # SM12x + fp32 routing: fp32 product, not bf16-exact, close to fp64.
    y, ref = _run_case(121, fp32, bf16, bf16)
    assert y.dtype == fp32
    assert (y != y.to(bf16).float()).any(), "logits are still bf16-rounded"
    assert (y.double() - ref).abs().max() < 1e-4
    # Today's tier 6 on the same inputs: bf16-exact, far from fp64.
    y6, _ = _run_case(121, fp32, bf16, bf16, env="0")
    assert torch.equal(y6, y6.to(bf16).float())
    assert (y6.double() - ref).abs().max() > (y.double() - ref).abs().max()
    # Other families, other out_dtypes and a biased gate keep tier 6.
    for case in ((100, fp32, bf16, bf16, None), (90, fp32, bf16, bf16, None),
                 (121, None, bf16, bf16, None), (121, bf16, bf16, bf16, None),
                 (121, fp32, bf16, bf16, torch.zeros(288, dtype=bf16,
                                                     device=os.environ.get("GLM53_TEST_DEVICE", "cpu")))):
        family, out_dtype, wdtype, xdtype, bias = case
        y_other, _ = _run_case(family, out_dtype, wdtype, xdtype, bias)
        assert torch.equal(y_other.float(), y_other.float().to(bf16).float()), case
    return True


def test_real_target() -> bool:
    if os.environ.get("GLM53_REQUIRE_TARGET") != "1":
        return False
    text = P.read_text(encoding="utf-8")
    out, status = apply_text(text)
    assert status in ("applied", "skipped"), f"{P}: {status}"
    compile(out, str(P), "exec")
    return True


def main() -> int:
    test_apply_then_skip()
    test_missing_anchor_fails_closed()
    numerics = test_numerics()
    target = test_real_target()
    print(f"test_router_fp32_sm12x: OK (numerics={'run' if numerics else 'skipped'}, "
          f"real target={'checked' if target else 'not requested'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
