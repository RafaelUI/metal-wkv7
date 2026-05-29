import mlx.core as mx
import sys
sys.path.insert(0, "/Users/s/Develop/metal-wkv7")
from wkv7_train_metal import wkv7_chunk
from wkv7_custom import _py_fwd_chunk, _py_bwd_chunk

# Тест 1: маленький (B=1, H=2) — как в debug тесте
print("=== B=1, H=2 ===")
for B, H in [(1, 2), (2, 4)]:
    T, D = 32, 64
    mx.random.seed(7)
    r = mx.random.normal((B,T,H,D)).astype(mx.float32) * 0.3
    w = (mx.abs(mx.random.normal((B,T,H,D)))*0.1+0.85).astype(mx.float32)
    k = mx.random.normal((B,T,H,D)).astype(mx.float32) * 0.3
    v = mx.random.normal((B,T,H,D)).astype(mx.float32) * 0.3
    a = mx.random.normal((B,T,H,D)).astype(mx.float32) * 0.1
    b = mx.random.normal((B,T,H,D)).astype(mx.float32) * 0.1
    h0 = mx.zeros((B,H,D,D))

    # Python backward reference
    _, _, h_all, sa_all = _py_fwd_chunk(r,w,k,v,a,b,h0)
    d1 = mx.ones((B,T,H,D), dtype=mx.float32)
    d0 = mx.zeros((B,H,D,D), dtype=mx.float32)
    dr_py,_,_,_,_,_,_ = _py_bwd_chunk(r,w,k,v,a,b,h_all,sa_all,d1,d0)
    mx.eval(dr_py)

    # Metal через wkv7_chunk
    def loss_fn(r_,w_,k_,v_,a_,b_):
        out, h, _ = wkv7_chunk(r_,w_,k_,v_,a_,b_,h0)
        return out.sum()

    _, gm = mx.value_and_grad(loss_fn)(r,w,k,v,a,b)
    mx.eval(gm)

    diff = mx.max(mx.abs(gm - dr_py)).item()
    print(f"B={B} H={H}: dr diff={diff:.6f} {'OK' if diff<1e-3 else 'FAIL'}")
