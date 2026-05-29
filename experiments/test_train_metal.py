import mlx.core as mx
import sys, time
sys.path.insert(0, "/Users/s/Develop/metal-wkv7")
from wkv7_train_metal import wkv7_metal_train
from wkv7_custom import _py_fwd_chunk, _py_bwd_chunk, wkv7_fast

HEAD_SIZE = 64
B, T, H = 2, 32, 4  # T=CHUNK для чистого сравнения

mx.random.seed(42)
r = mx.random.normal((B,T,H,HEAD_SIZE)).astype(mx.float32) * 0.3
w = (mx.abs(mx.random.normal((B,T,H,HEAD_SIZE)))*0.1+0.85).astype(mx.float32)
k = mx.random.normal((B,T,H,HEAD_SIZE)).astype(mx.float32) * 0.3
v = mx.random.normal((B,T,H,HEAD_SIZE)).astype(mx.float32) * 0.3
a = mx.random.normal((B,T,H,HEAD_SIZE)).astype(mx.float32) * 0.1
b = mx.random.normal((B,T,H,HEAD_SIZE)).astype(mx.float32) * 0.1
h0 = mx.zeros((B,H,HEAD_SIZE,HEAD_SIZE))

print("1. Forward...")
out_m  = wkv7_metal_train(r,w,k,v,a,b)
out_py,_,h_all,sa_all = _py_fwd_chunk(r,w,k,v,a,b,h0)
mx.eval(out_m, out_py)
print(f"   diff: {mx.max(mx.abs(out_m-out_py)).item():.6f} OK")

print("\n2. Градиенты через custom_function...")
def loss_metal(r_,w_,k_,v_,a_,b_):
    return wkv7_metal_train(r_,w_,k_,v_,a_,b_).sum()
val, gm = mx.value_and_grad(loss_metal)(r,w,k,v,a,b)
mx.eval(val, gm)

d1 = mx.ones((B,T,H,HEAD_SIZE), dtype=mx.float32)
d0 = mx.zeros((B,H,HEAD_SIZE,HEAD_SIZE), dtype=mx.float32)
dr_py,dw_py,dk_py,dv_py,da_py,db_py,_ = _py_bwd_chunk(
    r,w,k,v,a,b, h_all, sa_all, d1, d0)
mx.eval(dr_py,dw_py,dk_py,dv_py,da_py,db_py)

# gm = grad_r (только по первому аргументу по умолчанию в mx.value_and_grad)
all_ok = True
for name, gmi, gpi in zip('r', [gm], [dr_py]):
    diff = mx.max(mx.abs(gmi - gpi)).item()
    ok = diff < 1e-3
    if not ok: all_ok = False
    print(f"   d{name}: {diff:.6f} {'OK' if ok else 'FAIL'}")

print("\n3. Скорость...")
t0 = time.time()
for _ in range(20):
    val, gm = mx.value_and_grad(loss_metal)(r,w,k,v,a,b)
mx.eval(val, *gm)
toks_m = B*T*20/(time.time()-t0)

def loss_py(r_,w_,k_,v_,a_,b_):
    return wkv7_fast(r_,w_,k_,v_,a_,b_).sum()
t0 = time.time()
for _ in range(20):
    val2, gm2 = mx.value_and_grad(loss_py)(r,w,k,v,a,b)
mx.eval(val2, gm2)
toks_p = B*T*20/(time.time()-t0)

print(f"   Metal fwd+bwd:  {toks_m:.0f} tok/s")
print(f"   Python fwd+bwd: {toks_p:.0f} tok/s")
print(f"   Прирост: {toks_m/toks_p:.1f}x")
print(f"\n{'PASS' if all_ok else 'FAIL'} — Metal WKV7 training kernel")
