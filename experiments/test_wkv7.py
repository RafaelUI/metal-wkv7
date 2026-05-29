import mlx.core as mx
import sys, time
sys.path.insert(0, "/Users/s/Develop/metal-wkv7")
from wkv7_metal import wkv7_metal, HEAD_SIZE, CHUNK

def wkv7_reference(r, w, k, v, a, b, h):
    B, T, H, D = r.shape
    outs = []
    for t in range(T):
        r_t = r[:, t]; w_t = w[:, t]
        k_t = k[:, t]; v_t = v[:, t]
        a_t = a[:, t]; b_t = b[:, t]
        sa  = mx.einsum("bhsd,bhd->bhs", h, a_t)
        sab = mx.einsum("bhs,bhd->bhsd", sa, b_t)
        vk  = mx.einsum("bhs,bhd->bhsd", v_t, k_t)
        h   = h * w_t[:, :, None, :] + vk + sab
        outs.append(mx.einsum("bhsd,bhd->bhs", h, r_t))
    return mx.stack(outs, axis=1), h

B, H, D, T = 2, 4, HEAD_SIZE, CHUNK
mx.random.seed(42)
r  = mx.random.normal((B, T, H, D)).astype(mx.float32)
w  = (mx.abs(mx.random.normal((B, T, H, D))) * 0.1 + 0.9).astype(mx.float32)
k  = mx.random.normal((B, T, H, D)).astype(mx.float32)
v  = mx.random.normal((B, T, H, D)).astype(mx.float32)
a  = mx.random.normal((B, T, H, D)).astype(mx.float32) * 0.1
b  = mx.random.normal((B, T, H, D)).astype(mx.float32) * 0.1
h0 = mx.zeros((B, H, D, D), dtype=mx.float32)

print("Reference (Python)...")
out_ref, h_ref = wkv7_reference(r, w, k, v, a, b, h0)
mx.eval(out_ref, h_ref)

print("Metal kernel...")
out_m, h_m = wkv7_metal(r, w, k, v, a, b, h0)
mx.eval(out_m, h_m)

diff_out = mx.max(mx.abs(out_m - out_ref)).item()
diff_h   = mx.max(mx.abs(h_m   - h_ref  )).item()
print(f"Max diff out: {diff_out:.6f}")
print(f"Max diff h:   {diff_h:.6f}")

if diff_out < 1e-3 and diff_h < 1e-3:
    print("CORRECT")
    t0 = time.time()
    for _ in range(50):
        out_m, h_m = wkv7_metal(r, w, k, v, a, b, h0)
    mx.eval(out_m)
    dt = time.time() - t0
    print(f"Metal:  {B * T * 50 / dt:.0f} tok/s")
    t0 = time.time()
    for _ in range(50):
        out_ref, _ = wkv7_reference(r, w, k, v, a, b, h0)
    mx.eval(out_ref)
    dt = time.time() - t0
    print(f"Python: {B * T * 50 / dt:.0f} tok/s")
else:
    print("ERROR")
