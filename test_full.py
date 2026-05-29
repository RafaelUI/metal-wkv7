"""
test_full.py — Проверка корректности и скорости full-sequence kernel.
"""
import sys, os, time, statistics
sys.path.insert(0, os.path.expanduser("~/Develop/metal-wkv7"))
sys.path.insert(0, os.path.expanduser("~/Develop/rwkv-mlx"))

import mlx.core as mx
from wkv7_full import wkv7_full_train, CHUNK
from model.wkv7 import wkv7_train_py, wkv7_train as wkv7_chunked

def check(name, got, exp, tol=2e-4):
    diff = mx.max(mx.abs(got - exp)).item()
    status = "OK" if diff < tol else f"FAIL (diff={diff:.2e})"
    print(f"  {name}: {diff:.6f} {status}")
    return diff < tol

print("=" * 55)
print("test_full.py: full-sequence WKV7 Metal kernel")
print("=" * 55)

B, T, H, D = 2, 64, 4, 64   # 2 чанка (T=64, CHUNK=32)
mx.random.seed(42)
r = mx.random.normal((B,T,H,D))
w = mx.ones((B,T,H,D)) * 0.95
k = mx.random.normal((B,T,H,D)) * 0.3
v = mx.random.normal((B,T,H,D)) * 0.3
a = mx.random.normal((B,T,H,D)) * 0.1
b = mx.random.normal((B,T,H,D)) * 0.1

print("\n1. Forward (vs Python einsum)...")
out_py   = wkv7_train_py(r,w,k,v,a,b)
out_full = wkv7_full_train(r,w,k,v,a,b)
mx.eval(out_py, out_full)
ok_fwd = check("forward", out_full, out_py)

print("\n2. Forward (vs chunked Metal)...")
out_chk = wkv7_chunked(r,w,k,v,a,b)
mx.eval(out_chk)
ok_chk = check("vs chunked", out_full, out_chk)

print("\n3. Градиенты (vs Python einsum)...")
def fn_full(r,w,k,v,a,b): return mx.mean(wkv7_full_train(r,w,k,v,a,b))
def fn_py(r,w,k,v,a,b):   return mx.mean(wkv7_train_py(r,w,k,v,a,b))
_, gf = mx.value_and_grad(fn_full, argnums=list(range(6)))(r,w,k,v,a,b)
_, gp = mx.value_and_grad(fn_py,   argnums=list(range(6)))(r,w,k,v,a,b)
mx.eval(*gf, *gp)
ok_grad = all(check(f"d{n}", gf[i], gp[i]) for i, n in enumerate("rwkvab"))

print("\n4. Тест длиной T=512 (16 чанков)...")
T2 = 512
r2 = mx.random.normal((B,T2,H,D)); w2 = mx.ones((B,T2,H,D))*0.95
k2 = mx.random.normal((B,T2,H,D))*0.3; v2 = mx.random.normal((B,T2,H,D))*0.3
a2 = mx.random.normal((B,T2,H,D))*0.1; b2 = mx.random.normal((B,T2,H,D))*0.1
out_py2   = wkv7_train_py(r2,w2,k2,v2,a2,b2)
out_full2 = wkv7_full_train(r2,w2,k2,v2,a2,b2)
mx.eval(out_py2, out_full2)
ok_t512 = check("fwd T=512", out_full2, out_py2)

def fn_f2(r,w,k,v,a,b): return mx.mean(wkv7_full_train(r,w,k,v,a,b))
def fn_p2(r,w,k,v,a,b): return mx.mean(wkv7_train_py(r,w,k,v,a,b))
_, gf2 = mx.value_and_grad(fn_f2, argnums=list(range(6)))(r2,w2,k2,v2,a2,b2)
_, gp2 = mx.value_and_grad(fn_p2, argnums=list(range(6)))(r2,w2,k2,v2,a2,b2)
mx.eval(*gf2, *gp2)
ok_g512 = all(check(f"d{n} T=512", gf2[i], gp2[i]) for i, n in enumerate("rwkvab"))

print("\n5. Скорость (B=2, T=512, H=4, D=64)...")
# Прогрев обоих ядер вместе
for _ in range(10):
    mx.eval(wkv7_full_train(r2,w2,k2,v2,a2,b2))
    mx.eval(wkv7_chunked(r2,w2,k2,v2,a2,b2))
time.sleep(0.5)

def bench(fn, n=80):
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        mx.eval(fn())
        times.append(time.perf_counter() - t0)
    return statistics.median(times)

t_full = bench(lambda: wkv7_full_train(r2,w2,k2,v2,a2,b2))
t_chk  = bench(lambda: wkv7_chunked(r2,w2,k2,v2,a2,b2))
tok_s_full = B * T2 / t_full
tok_s_chk  = B * T2 / t_chk
print(f"  Full-seq kernel:  {tok_s_full:,.0f} tok/s  ({t_full*1000:.2f} ms)")
print(f"  Chunked kernel:   {tok_s_chk:,.0f} tok/s  ({t_chk*1000:.2f} ms)")
print(f"  Соотношение: {tok_s_full/tok_s_chk:.2f}×")

print("\n6. Скорость fwd+bwd (интеграция в шаг обучения)...")
def fwd_bwd_full():
    loss, grads = mx.value_and_grad(
        lambda r,w,k,v,a,b: mx.mean(wkv7_full_train(r,w,k,v,a,b)),
        argnums=list(range(6))
    )(r2,w2,k2,v2,a2,b2)
    mx.eval(loss, *grads)
    return loss

def fwd_bwd_chk():
    loss, grads = mx.value_and_grad(
        lambda r,w,k,v,a,b: mx.mean(wkv7_chunked(r,w,k,v,a,b)),
        argnums=list(range(6))
    )(r2,w2,k2,v2,a2,b2)
    mx.eval(loss, *grads)
    return loss

for _ in range(10):
    fwd_bwd_full(); fwd_bwd_chk()
time.sleep(0.5)
t_fb_full = bench(fwd_bwd_full)
t_fb_chk  = bench(fwd_bwd_chk)
tok_fb_full = B * T2 / t_fb_full
tok_fb_chk  = B * T2 / t_fb_chk
print(f"  Full-seq fwd+bwd: {tok_fb_full:,.0f} tok/s  ({t_fb_full*1000:.2f} ms)")
print(f"  Chunked fwd+bwd:  {tok_fb_chk:,.0f} tok/s  ({t_fb_chk*1000:.2f} ms)")
print(f"  Соотношение: {tok_fb_full/tok_fb_chk:.2f}×")

all_ok = ok_fwd and ok_chk and ok_grad and ok_t512 and ok_g512
print("\n" + ("=" * 55))
print("РЕЗУЛЬТАТ:", "PASS ✓" if all_ok else "FAIL ✗")
print("=" * 55)
