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
h_T_py = h_all[T]; sa_py = mx.stack(sa_all, axis=1)
mx.eval(h_T_py, sa_py)

debug = {}

@mx.custom_function
def chunk_dbg(r_,w_,k_,v_,a_,b_,h_in_):
    res = _get_fwd(H)(
        inputs=[x.astype(mx.float32) for x in [r_,w_,k_,v_,a_,b_,h_in_]],
        grid=(B*H,D,1), threadgroup=(1,1,1),
        output_shapes=[(B,T,H,D),(B,H,D,D),(B,T,H,D)],
        output_dtypes=[mx.float32]*3,
    )
    return res[0], res[1], res[2]

@chunk_dbg.vjp
def chunk_dbg_vjp(primals, cotangents, outputs):
    r_,w_,k_,v_,a_,b_,h_in_ = primals
    d_out_, d_h_, _           = cotangents
    _,      h_out_, sa_out_   = outputs
    mx.eval(d_out_, d_h_, h_out_, sa_out_)

    debug['d_out_norm']  = mx.sum(d_out_**2).item()**.5
    debug['d_h_norm']    = mx.sum(d_h_**2).item()**.5
    debug['h_out_diff']  = mx.max(mx.abs(h_out_ - h_T_py)).item()
    debug['sa_out_diff'] = mx.max(mx.abs(sa_out_ - sa_py)).item()
    debug['h_out_shape'] = str(h_out_.shape)
    debug['sa_shape']    = str(sa_out_.shape)

    res = _get_bwd(H)(
        inputs=[r_,w_,k_,v_,a_,b_,h_out_,sa_out_,d_out_,d_h_],
        grid=(B*H,D,1), threadgroup=(1,1,1),
        output_shapes=[(B,T,H,D)]*6+[(B,H,D,D)],
        output_dtypes=[mx.float32]*7,
        init_value=0,
    )
    return res[0],res[1],res[2],res[3],res[4],res[5],res[6]

def loss_fn(r_,w_,k_,v_,a_,b_):
    out,h,_ = chunk_dbg(r_,w_,k_,v_,a_,b_,h0)
    return out.sum()

_, gm = mx.value_and_grad(loss_fn)(r,w,k,v,a,b)
mx.eval(*gm)

print("VJP debug (B=2,H=4):")
for k_,v_ in debug.items():
    print(f"  {k_}: {v_}")

d1 = mx.ones((B,T,H,D), dtype=mx.float32)
d0 = mx.zeros((B,H,D,D), dtype=mx.float32)
dr_py,dw_py,dk_py,dv_py,da_py,db_py,_ = _py_bwd_chunk(
    r,w,k,v,a,b,h_all,sa_all,d1,d0)
mx.eval(dr_py)

print("\nГрадиенты:")
for name,gi,gp in zip('rwkvab',gm,[dr_py,dw_py,dk_py,dv_py,da_py,db_py]):
    diff = mx.max(mx.abs(gi-gp)).item()
    print(f"  d{name}: {diff:.6f} {'OK' if diff<1e-3 else 'FAIL'}")
