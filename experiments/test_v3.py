import mlx.core as mx, sys, time
sys.path.insert(0, '/Users/s/Develop/metal-wkv7')
from wkv7_bwd_v3    import _get_bwd_v3
from wkv7_train_metal import wkv7_metal_train   # v2 для сравнения
from wkv7_custom    import _py_fwd_chunk, _py_bwd_chunk

HEAD_SIZE = 64
CHUNK = 32

# ─── Сборка v3 через custom_function ────────────────────────────────────────
from wkv7_train_metal import _get_fwd  # forward тот же

@mx.custom_function
def _chunk_v3(r, w, k, v, a, b, h_in):
    B, T, H, D = r.shape
    from wkv7_train_metal import _get_fwd
    res = _get_fwd(H)(
        inputs=[x.astype(mx.float32) for x in [r,w,k,v,a,b,h_in]],
        grid=(B*H, D, 1), threadgroup=(1,1,1),
        output_shapes=[(B,T,H,D),(B,H,D,D),(B,T,H,D)],
        output_dtypes=[mx.float32]*3,
    )
    return res[0], res[1], res[2]

@_chunk_v3.vjp
def _chunk_v3_vjp(primals, cotangents, outputs):
    r,w,k,v,a,b,h_in = primals
    d_out,d_h_out,_   = cotangents
    _,h_fwd,sa_fwd    = outputs
    mx.eval(h_fwd, sa_fwd, d_out, d_h_out)
    B,T,H,D = r.shape
    res = _get_bwd_v3(H)(
        inputs=[x.astype(mx.float32) for x in [r,w,k,v,a,b,h_fwd,sa_fwd,d_out,d_h_out]],
        grid=(B*H*D,1,1), threadgroup=(D,1,1),
        output_shapes=[(B,T,H,D)]*6+[(B,H,D,D)],
        output_dtypes=[mx.float32]*7,
    )
    return res[0],res[1],res[2],res[3],res[4],res[5],res[6]

def wkv7_v3(r, w, k, v, a, b):
    B,T,H,D = r.shape
    h = mx.zeros((B,H,D,D))
    outs = []
    for s in range(0, T, CHUNK):
        e = min(s+CHUNK, T); cl = e-s
        rc,wc,kc,vc,ac,bc = (x[:,s:e] for x in (r,w,k,v,a,b))
        if cl < CHUNK:
            pad = CHUNK-cl
            def p(x,val=0.):
                return mx.pad(x,[(0,0),(0,pad),(0,0),(0,0)],constant_values=val)
            rc=p(rc);wc=p(wc,1.);kc=p(kc);vc=p(vc);ac=p(ac);bc=p(bc)
        out_c,h,_ = _chunk_v3(rc,wc,kc,vc,ac,bc,h)
        mx.eval(h, out_c)
        outs.append(out_c[:,:cl])
    return mx.concatenate(outs, axis=1)

# ─── Тест ────────────────────────────────────────────────────────────────────
B,T,H,D = 2,32,4,64
mx.random.seed(42)
r=mx.random.normal((B,T,H,D)).astype(mx.float32)*0.3
w=(mx.abs(mx.random.normal((B,T,H,D)))*0.1+0.85).astype(mx.float32)
k=mx.random.normal((B,T,H,D)).astype(mx.float32)*0.3
v=mx.random.normal((B,T,H,D)).astype(mx.float32)*0.3
a=mx.random.normal((B,T,H,D)).astype(mx.float32)*0.1
b=mx.random.normal((B,T,H,D)).astype(mx.float32)*0.1
h0=mx.zeros((B,H,D,D))

print("1. Градиенты v3 vs Python reference...")
_,_,h_all,sa_all = _py_fwd_chunk(r,w,k,v,a,b,h0)
d1=mx.ones((B,T,H,D),dtype=mx.float32)
d0=mx.zeros((B,H,D,D),dtype=mx.float32)
dr_py,dw_py,dk_py,dv_py,da_py,db_py,_ = _py_bwd_chunk(r,w,k,v,a,b,h_all,sa_all,d1,d0)
mx.eval(dr_py,dw_py,dk_py,dv_py,da_py,db_py)

def l3(r_,w_,k_,v_,a_,b_): return wkv7_v3(r_,w_,k_,v_,a_,b_).sum()
def l2(r_,w_,k_,v_,a_,b_): return wkv7_metal_train(r_,w_,k_,v_,a_,b_).sum()

_,g3=mx.value_and_grad(l3,argnums=[0,1,2,3,4,5])(r,w,k,v,a,b)
_,g2=mx.value_and_grad(l2,argnums=[0,1,2,3,4,5])(r,w,k,v,a,b)
mx.eval(*g3,*g2)

names=['r','w','k','v','a','b']
refs =[dr_py,dw_py,dk_py,dv_py,da_py,db_py]
all_ok=True
for nm,gv3,gv2,gp in zip(names,g3,g2,refs):
    d_py=mx.max(mx.abs(gv3-gp)).item()
    d_v2=mx.max(mx.abs(gv3-gv2)).item()
    ok=d_py<1e-3
    if not ok: all_ok=False
    print(f"   d{nm}: vs py={d_py:.2e}{'✓' if ok else '✗'}  vs v2={d_v2:.2e}")

print("\n2. Скорость fwd+bwd (30 прогонов)...")
N=30
for fn,name in [(l3,'v3'),(l2,'v2')]:
    for _ in range(3):
        _,g=mx.value_and_grad(fn,argnums=[0,1,2,3,4,5])(r,w,k,v,a,b); mx.eval(*g)
    t0=time.perf_counter()
    for _ in range(N):
        _,g=mx.value_and_grad(fn,argnums=[0,1,2,3,4,5])(r,w,k,v,a,b); mx.eval(*g)
    toks=B*T*N/(time.perf_counter()-t0)
    print(f"   {name}: {toks:.0f} tok/s")

print(f"\n{'PASS ✓' if all_ok else 'FAIL ✗'}")
