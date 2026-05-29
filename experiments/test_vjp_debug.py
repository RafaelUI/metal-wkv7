import mlx.core as mx
import sys
sys.path.insert(0, "/Users/s/Develop/metal-wkv7")
from wkv7_custom import _py_fwd_chunk
from wkv7_train_metal import _get_fwd, _get_bwd, HEAD_SIZE, CHUNK

B, T, H, D = 1, CHUNK, 2, HEAD_SIZE
mx.random.seed(7)
r = mx.random.normal((B,T,H,D)).astype(mx.float32) * 0.3
w = (mx.abs(mx.random.normal((B,T,H,D)))*0.1+0.85).astype(mx.float32)
k = mx.random.normal((B,T,H,D)).astype(mx.float32) * 0.3
v = mx.random.normal((B,T,H,D)).astype(mx.float32) * 0.3
a = mx.random.normal((B,T,H,D)).astype(mx.float32) * 0.1
b = mx.random.normal((B,T,H,D)).astype(mx.float32) * 0.1
h0 = mx.zeros((B,H,D,D))

# Эталонные значения из Python forward
out_py, _, h_all, sa_all = _py_fwd_chunk(r, w, k, v, a, b, h0)
h_T_py   = h_all[T]
sa_py    = mx.stack(sa_all, axis=1)
mx.eval(out_py, h_T_py, sa_py)

# Версия wkv7_chunk с отладкой внутри VJP
debug_info = {}

@mx.custom_function
def wkv7_chunk_debug(r, w, k, v, a, b, h_in):
    res = _get_fwd(H)(
        inputs=[x.astype(mx.float32) for x in [r,w,k,v,a,b,h_in]],
        grid=(B*H, D, 1), threadgroup=(1,1,1),
        output_shapes=[(B,T,H,D), (B,H,D,D), (B,T,H,D)],
        output_dtypes=[mx.float32]*3,
    )
    return res[0], res[1], res[2]

@wkv7_chunk_debug.vjp
def wkv7_chunk_debug_vjp(primals, cotangents, outputs):
    r_, w_, k_, v_, a_, b_, h_in_ = primals
    d_out_, d_h_out_, d_sa_     = cotangents
    out_,   h_out_,  sa_out_    = outputs

    mx.eval(d_out_, d_h_out_, h_out_, sa_out_)

    # Записываем для анализа
    debug_info['d_out_shape']  = d_out_.shape
    debug_info['d_out_norm']   = mx.sum(d_out_**2).item()**.5
    debug_info['d_h_norm']     = mx.sum(d_h_out_**2).item()**.5
    debug_info['h_out_diff']   = mx.max(mx.abs(h_out_ - h_T_py)).item()
    debug_info['sa_out_diff']  = mx.max(mx.abs(sa_out_ - sa_py)).item()

    res = _get_bwd(H)(
        inputs=[r_,w_,k_,v_,a_,b_,h_out_,sa_out_,d_out_,d_h_out_],
        grid=(B*H, D, 1), threadgroup=(1,1,1),
        output_shapes=[(B,T,H,D)]*6+[(B,H,D,D)],
        output_dtypes=[mx.float32]*7,
        init_value=0,
    )
    return res[0],res[1],res[2],res[3],res[4],res[5],res[6]

def loss_fn(r_,w_,k_,v_,a_,b_):
    out, h, _ = wkv7_chunk_debug(r_,w_,k_,v_,a_,b_,h0)
    return out.sum()

_, grads = mx.value_and_grad(loss_fn)(r,w,k,v,a,b)
mx.eval(*grads)

print("Что приходит в VJP:")
for k_, v_ in debug_info.items():
    print(f"  {k_}: {v_}")

# Сравниваем с Python backward
from wkv7_custom import _py_bwd_chunk
d_out_ref = mx.ones((B,T,H,D), dtype=mx.float32)
d_h_ref   = mx.zeros((B,H,D,D), dtype=mx.float32)
dr_py,dw_py,dk_py,dv_py,da_py,db_py,_ = _py_bwd_chunk(
    r,w,k,v,a,b, h_all, sa_all, d_out_ref, d_h_ref)
mx.eval(dr_py,dw_py,dk_py,dv_py,da_py,db_py)

print("\nСравнение градиентов:")
for name, gm, gp in zip(['r','w','k','v','a','b'],
    grads, [dr_py,dw_py,dk_py,dv_py,da_py,db_py]):
    diff = mx.max(mx.abs(gm - gp)).item()
    print(f"  d{name}: diff={diff:.6f} {'OK' if diff<1e-3 else 'FAIL'}")
