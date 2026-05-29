# RWKV-7 Metal Backward Kernel for Apple MLX

Standalone Metal kernel implementing WKV-7 forward and backward pass
for Apple Silicon via MLX. Drop-in replacement for Python-based chunked implementation.

**7.8× faster than Python einsum baseline. 1.73× faster than chunked Metal approach.**

## What's inside

```
wkv7_checkpoint.py  ← main kernel (forward + backward, mx.custom_function VJP)
wkv7_custom.py      ← Python reference implementation (correctness baseline)
wkv7_metal.py       ← inference-only kernel
test_full.py        ← correctness + speed tests
test_isolate.py     ← isolated kernel tests
experiments/        ← documented failed attempts (simd_sum, bank padding)
HANDOFF.md          ← full development log, math, profiling results
```

## Usage

```python
from wkv7_checkpoint import make_wkv7_checkpoint

# Create once per (B, T, H) — JIT compiled
wkv7 = make_wkv7_checkpoint(B=4, T=1024, H=6, D=64)

# Forward + backward (mx.custom_function, VJP registered)
output = wkv7(r, w, k, v, a, b)   # → (B, T, H, D), bf16 or fp32
```

Compatible with `mx.compile` and bf16 training.

## Results (M4 Air 16GB, debug 36.4M model)

| Version | tok/s | vs Python |
|---------|-------|-----------|
| Python einsum | ~900 | 1× |
| Metal v2 chunked | 3 666 | 4.1× |
| **This kernel** | **~6 978** | **7.8×** |

## Key design: checkpoint approach

The backward pass reconstructs `h_prev = (h_cur - v*k - sa*b) / w`.
This amplifies errors by `(1/w)^N`. With N=512: overflow. With CHUNK=32: ×30, stable.

Solution: save `h_checkpoints[c]` every 32 tokens in forward pass,
load exact checkpoints in backward pass — O(N_CHUNKS × D²) memory.

## See also

[rwkv-mlx](https://github.com/yourusername/rwkv-mlx) — full pretraining pipeline using this kernel.
