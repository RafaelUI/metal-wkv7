import mlx.core as mx
import numpy as np
import sys, time
sys.path.insert(0, "/Users/s/Develop/metal-wkv7")
from wkv7_custom import wkv7_fast, _py_fwd_chunk

HEAD_SIZE = 64
B, T, H = 2, 64, 4

mx.random.seed(42)
r = mx.random.normal((B, T, H, HEAD_SIZE)).astype(mx.float32) * 0.3
w = (mx.abs(mx.random.normal((B, T, H, HEAD_SIZE))) * 0.1 + 0.85).astype(mx.float32)
k = mx.random.normal((B, T, H, HEAD_SIZE)).astype(mx.float32) * 0.3
v = mx.random.normal((B, T, H, HEAD_SIZE)).astype(mx.float32) * 0.3
a = mx.random.normal((B, T, H, HEAD_SIZE)).astype(mx.float32) * 0.1
b = mx.random.normal((B, T, H, HEAD_SIZE)).astype(mx.float32) * 0.1

print("1. Проверка forward (Metal vs Python)...")
out_metal = wkv7_fast(r, w, k, v, a, b)
mx.eval(out_metal)

h0 = mx.zeros((B, H, HEAD_SIZE, HEAD_SIZE))
out_py, _, _, _ = _py_fwd_chunk(r[:, :32], w[:, :32], k[:, :32],
                                  v[:, :32], a[:, :32], b[:, :32], h0)
mx.eval(out_py)
diff = mx.max(mx.abs(out_metal[:, :32] - out_py)).item()
print(f"   max_diff forward: {diff:.6f} {'OK' if diff < 1e-4 else 'FAIL'}")

print("2. Проверка градиентов через mx.value_and_grad...")
def loss_fn(r_, w_, k_, v_, a_, b_):
    out = wkv7_fast(r_, w_, k_, v_, a_, b_)
    return out.sum()

val, grads = mx.value_and_grad(loss_fn)(r, w, k, v, a, b)
mx.eval(val, *grads)
print(f"   loss: {val.item():.4f}")
has_nan = any(mx.any(mx.isnan(g)).item() for g in grads)
print(f"   NaN в градиентах: {has_nan}")
print(f"   grad_r norm: {mx.sum(grads[0]**2).item()**.5:.4f}")
print(f"   {'OK — градиенты работают!' if not has_nan else 'FAIL'}")

print("\n3. Замер скорости...")
t0 = time.time()
for _ in range(20):
    val, grads = mx.value_and_grad(loss_fn)(r, w, k, v, a, b)
mx.eval(val, *grads)
dt = time.time() - t0
tok_s = B * T * 20 / dt
print(f"   wkv7_fast (fwd+bwd): {tok_s:.0f} tok/s")

# Сравнение с чистым Python
from wkv7_custom import _py_fwd_chunk, _py_bwd_chunk
def loss_py(r_, w_, k_, v_, a_, b_):
    h0_ = mx.zeros((B, H, HEAD_SIZE, HEAD_SIZE))
    out_list = []
    for s in range(0, T, 32):
        o, h0_, _, _ = _py_fwd_chunk(r_[:,s:s+32], w_[:,s:s+32], k_[:,s:s+32],
                                      v_[:,s:s+32], a_[:,s:s+32], b_[:,s:s+32], h0_)
        out_list.append(o)
    return mx.concatenate(out_list, axis=1).sum()

t0 = time.time()
for _ in range(20):
    val2, grads2 = mx.value_and_grad(loss_py)(r, w, k, v, a, b)
mx.eval(val2, *grads2)
dt2 = time.time() - t0
tok_s2 = B * T * 20 / dt2
print(f"   wkv7_python (fwd+bwd): {tok_s2:.0f} tok/s")
print(f"   Прирост: {tok_s/tok_s2:.1f}x")
