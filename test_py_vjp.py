import mlx.core as mx
import sys
sys.path.insert(0, "/Users/s/Develop/metal-wkv7")
from wkv7_train_metal import _get_fwd, _get_bwd, HEAD_SIZE, CHUNK
from wkv7_custom import _py_fwd_chunk, _py_bwd_chunk

B, T, H, D = 2, CHUNK, 4, HEAD_SIZE
mx.random.seed(7)
r = mx.random.normal((B,T,H,D)).astype(mx.float32) * 0.3
w = (mx.abs(mx.random.normal((B,T,H,D)))*0.1+0.85).astype(mx.float32)
k = mx.random.normal((B,T,H,D)).astype(mx.float32) * 0.3
v = mx.random.normal((B,T,H,D)).astype(mx.float32) * 0.3
a = mx.random.normal((B,T,H,D)).astype(mx.float32) * 0.1
b = mx.random.normal((B,T,H,D)).astype(mx.float32) * 0.1
h0 = mx.zeros((B,H,D,D))

_, _, h_all, sa_all = _py_fwd_chunk(r,w,k,v,a,b,h0)
d1 = mx.ones((B,T,H,D), dtype=mx.float32)
d0 = mx.zeros((B,H,D,D), dtype=mx.float32)
dr_py,dw_py,dk_py,dv_py,da_py,db_py,_ = _py_bwd_chunk(r,w,k,v,a,b,h_all,sa_all,d1,d0)
mx.eval(dr_py)

print("Metal backward НАПРЯМУЮ (B=2, H=4):")
h_T = h_all[T]; sa = mx.stack(sa_all, axis=1)
mx.eval(h_T, sa)
res = _get_bwd(H)(
    inputs=[r,w,k,v,a,b,h_T,sa,d1,d0],
    grid=(B*H,D,1), threadgroup=(1,1,1),
    output_shapes=[(B,T,H,D)]*6+[(B,H,D,D)],
    output_dtypes=[mx.float32]*7,
    init_value=0,
)
mx.eval(*res)
for name, ri, gp in zip('rwkvab', res, [dr_py,dw_py,dk_py,dv_py,da_py,db_py]):
    diff = mx.max(mx.abs(ri - gp)).item()
    print(f"  d{name}: {diff:.6f} {'OK' if diff<1e-3 else 'FAIL'}")
