import mlx.core as mx
import sys
sys.path.insert(0, "/Users/s/Develop/metal-wkv7")
from wkv7_train_metal import _get_fwd, _get_bwd, HEAD_SIZE, CHUNK
from wkv7_custom import _py_fwd_chunk, _py_bwd_chunk

B, T, H, D = 1, CHUNK, 2, HEAD_SIZE
mx.random.seed(7)
r = mx.random.normal((B,T,H,D)).astype(mx.float32) * 0.3
w = (mx.abs(mx.random.normal((B,T,H,D)))*0.1+0.85).astype(mx.float32)
k = mx.random.normal((B,T,H,D)).astype(mx.float32) * 0.3
v = mx.random.normal((B,T,H,D)).astype(mx.float32) * 0.3
a = mx.random.normal((B,T,H,D)).astype(mx.float32) * 0.1
b = mx.random.normal((B,T,H,D)).astype(mx.float32) * 0.1
h0 = mx.zeros((B,H,D,D))
d_out = mx.ones((B,T,H,D), dtype=mx.float32)
d_h   = mx.zeros((B,H,D,D), dtype=mx.float32)

# Python forward — правильные значения
out_py, _, h_all, sa_all = _py_fwd_chunk(r, w, k, v, a, b, h0)
sa_py = mx.stack(sa_all, axis=1)
h_T   = h_all[T]  # финальное состояние
mx.eval(out_py, sa_py, h_T)

# Python backward — эталон
dr_py,dw_py,dk_py,dv_py,da_py,db_py,_ = _py_bwd_chunk(
    r, w, k, v, a, b, h_all, sa_all, d_out, d_h)
mx.eval(dr_py,dw_py,dk_py,dv_py,da_py,db_py)

print("Прямой вызов Metal backward kernel...")
bwd = _get_bwd(H)
res = bwd(
    inputs=[r,w,k,v,a,b, h_T.astype(mx.float32),
            sa_py.astype(mx.float32),
            d_out, d_h],
    grid=(B*H*D, 1, 1), threadgroup=(D,1,1),
    output_shapes=[(B,T,H,D)]*6 + [(B,H,D,D)],
    output_dtypes=[mx.float32]*7,
    init_value=0,
)
dr_m,dw_m,dk_m,dv_m,da_m,db_m,dh0_m = res
mx.eval(*res)

print(f"{'param':<6} {'max_diff':>12} {'py_max':>10} {'status':>6}")
print("-" * 38)
for name, mm, mp in zip(['r','w','k','v','a','b'],
    [dr_m,dw_m,dk_m,dv_m,da_m,db_m],
    [dr_py,dw_py,dk_py,dv_py,da_py,db_py]):
    diff = mx.max(mx.abs(mm - mp)).item()
    mx = mx  # just to keep mx in scope
    pmax = mx.max(mx.abs(mp)).item()
    print(f"{name:<6} {diff:>12.6f} {pmax:>10.4f} {'OK' if diff<1e-3 else 'FAIL':>6}")
