import mlx.core as mx, sys, time
sys.path.insert(0, "/Users/s/Develop/metal-wkv7")
from wkv7_simd        import wkv7_simd_train, _py_fwd_chunk, _py_bwd_chunk
from wkv7_train_metal import wkv7_metal_train

B, T, H, D = 2, 32, 4, 64
mx.random.seed(42)
r = mx.random.normal((B,T,H,D)).astype(mx.float32) * 0.3
w = (mx.abs(mx.random.normal((B,T,H,D)))*0.1 + 0.85).astype(mx.float32)
k = mx.random.normal((B,T,H,D)).astype(mx.float32) * 0.3
v = mx.random.normal((B,T,H,D)).astype(mx.float32) * 0.3
a = mx.random.normal((B,T,H,D)).astype(mx.float32) * 0.1
b = mx.random.normal((B,T,H,D)).astype(mx.float32) * 0.1
h0 = mx.zeros((B,H,D,D))

print("=" * 52)
print("  ТЕСТ wkv7_simd: simd_sum backward")
print("=" * 52)

print("\n1. Forward...")
out_simd = wkv7_simd_train(r,w,k,v,a,b)
out_v2   = wkv7_metal_train(r,w,k,v,a,b)
mx.eval(out_simd, out_v2)
df = mx.max(mx.abs(out_simd - out_v2)).item()
print(f"   vs v2: {df:.2e}  {'OK' if df < 1e-5 else 'FAIL'}")

print("\n2. Все 6 градиентов vs Python...")
_, _, h_all, sa_all = _py_fwd_chunk(r,w,k,v,a,b,h0)
d1 = mx.ones((B,T,H,D), dtype=mx.float32)
d0 = mx.zeros((B,H,D,D), dtype=mx.float32)
dr_py,dw_py,dk_py,dv_py,da_py,db_py,_ = _py_bwd_chunk(r,w,k,v,a,b,h_all,sa_all,d1,d0)
mx.eval(dr_py,dw_py,dk_py,dv_py,da_py,db_py)

def loss_simd(r_,w_,k_,v_,a_,b_): return wkv7_simd_train(r_,w_,k_,v_,a_,b_).sum()
def loss_v2(r_,w_,k_,v_,a_,b_):   return wkv7_metal_train(r_,w_,k_,v_,a_,b_).sum()

_, gs = mx.value_and_grad(loss_simd, argnums=[0,1,2,3,4,5])(r,w,k,v,a,b)
_, gv = mx.value_and_grad(loss_v2,   argnums=[0,1,2,3,4,5])(r,w,k,v,a,b)
mx.eval(*gs, *gv)

names = ['r','w','k','v','a','b']
refs  = [dr_py, dw_py, dk_py, dv_py, da_py, db_py]
all_ok = True
for nm, g_simd, g_v2, gp in zip(names, gs, gv, refs):
    diff_py = mx.max(mx.abs(g_simd - gp)).item()
    diff_v2 = mx.max(mx.abs(g_simd - g_v2)).item()
    ok = diff_py < 1e-3
    if not ok: all_ok = False
    print(f"   d{nm}:  vs py={diff_py:.2e} {'✓' if ok else '✗'}  vs v2={diff_v2:.2e}")

print("\n3. Скорость (30 прогонов fwd+bwd)...")
N = 30
for _ in range(3):
    _, g = mx.value_and_grad(loss_simd, argnums=[0,1,2,3,4,5])(r,w,k,v,a,b)
    mx.eval(*g)
for _ in range(3):
    _, g = mx.value_and_grad(loss_v2, argnums=[0,1,2,3,4,5])(r,w,k,v,a,b)
    mx.eval(*g)

t0 = time.perf_counter()
for _ in range(N):
    _, g = mx.value_and_grad(loss_simd, argnums=[0,1,2,3,4,5])(r,w,k,v,a,b)
    mx.eval(*g)
toks_s = B*T*N / (time.perf_counter() - t0)

t0 = time.perf_counter()
for _ in range(N):
    _, g = mx.value_and_grad(loss_v2, argnums=[0,1,2,3,4,5])(r,w,k,v,a,b)
    mx.eval(*g)
toks_v = B*T*N / (time.perf_counter() - t0)

print(f"   simd: {toks_s:8.0f} tok/s")
print(f"   v2:   {toks_v:8.0f} tok/s")
print(f"   Ускорение: {toks_s/toks_v:.2f}×")

print("\n" + "=" * 52)
print(f"  {'PASS ✓' if all_ok else 'FAIL ✗'}")
print("=" * 52)
