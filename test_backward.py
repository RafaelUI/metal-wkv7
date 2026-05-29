import mlx.core as mx
import numpy as np

HEAD_SIZE = 8
CHUNK     = 4

def wkv7_fwd(r, w, k, v, a, b, h):
    B, T, H, D = r.shape
    outs, h_all, sa_all = [], [h], []
    for t in range(T):
        r_t,w_t,k_t,v_t,a_t,b_t = r[:,t],w[:,t],k[:,t],v[:,t],a[:,t],b[:,t]
        sa  = mx.einsum("bhsd,bhd->bhs", h, a_t)
        h   = h * w_t[:,:,None,:] + mx.einsum("bhs,bhd->bhsd",v_t,k_t) + mx.einsum("bhs,bhd->bhsd",sa,b_t)
        outs.append(mx.einsum("bhsd,bhd->bhs", h, r_t))
        h_all.append(h); sa_all.append(sa)
    return mx.stack(outs, axis=1), h, h_all, sa_all

def wkv7_bwd(r, w, k, v, a, b, h_all, sa_all, d_out, d_h):
    B, T, H, D = r.shape
    C = d_h
    dr_list, dw_list, dk_list, dv_list, da_list, db_list = [], [], [], [], [], []

    for t in range(T-1, -1, -1):
        r_t,w_t,k_t,v_t,a_t,b_t = r[:,t],w[:,t],k[:,t],v[:,t],a[:,t],b[:,t]
        dy_t   = d_out[:, t]
        h_prev = h_all[t]
        h_cur  = h_all[t+1]
        sa_t   = sa_all[t]

        C   = C + mx.einsum("bhs,bhd->bhsd", dy_t, r_t)
        dsa = mx.einsum("bhsd,bhd->bhs", C, b_t)

        dr_list.insert(0, mx.einsum("bhs,bhsd->bhd", dy_t, h_cur))
        dw_list.insert(0, mx.sum(C * h_prev, axis=2))
        dk_list.insert(0, mx.einsum("bhsd,bhs->bhd", C, v_t))
        dv_list.insert(0, mx.einsum("bhsd,bhd->bhs", C, k_t))
        db_list.insert(0, mx.einsum("bhs,bhsd->bhd", sa_t, C))
        da_list.insert(0, mx.einsum("bhs,bhsd->bhd", dsa, h_prev))

        C = C * w_t[:,:,None,:] + mx.einsum("bhs,bhd->bhsd", dsa, a_t)

    dr = mx.stack(dr_list, axis=1)
    dw = mx.stack(dw_list, axis=1)
    dk = mx.stack(dk_list, axis=1)
    dv = mx.stack(dv_list, axis=1)
    da = mx.stack(da_list, axis=1)
    db = mx.stack(db_list, axis=1)
    return dr, dw, dk, dv, da, db, C

def num_grad(fn, x, eps=1e-3):
    x_np = np.array(x.tolist(), dtype=np.float64)
    g = np.zeros_like(x_np)
    for idx in np.ndindex(x_np.shape):
        orig = x_np[idx]
        x_np[idx] = orig + eps
        fp = fn(mx.array(x_np, dtype=mx.float32)).sum().item()
        x_np[idx] = orig - eps
        fm = fn(mx.array(x_np, dtype=mx.float32)).sum().item()
        x_np[idx] = orig
        g[idx] = (fp - fm) / (2 * eps)
    return mx.array(g.astype(np.float32))

B,T,H,D = 1,CHUNK,2,HEAD_SIZE
mx.random.seed(7)
r = mx.random.normal((B,T,H,D)) * 0.3
w = (mx.abs(mx.random.normal((B,T,H,D))) * 0.1 + 0.85).astype(mx.float32)
k = mx.random.normal((B,T,H,D)) * 0.3
v = mx.random.normal((B,T,H,D)) * 0.3
a = mx.random.normal((B,T,H,D)) * 0.1
b = mx.random.normal((B,T,H,D)) * 0.1
h0 = mx.zeros((B,H,D,D))
d_out = mx.random.normal((B,T,H,D)) * 0.3
d_h   = mx.zeros((B,H,D,D))

out, _, h_all, sa_all = wkv7_fwd(r, w, k, v, a, b, h0)
mx.eval(out)
dr,dw,dk,dv,da,db,dh0 = wkv7_bwd(r,w,k,v,a,b,h_all,sa_all,d_out,d_h)
mx.eval(dr,dw,dk,dv,da,db)

fns = {
    'r': lambda x: wkv7_fwd(x,w,k,v,a,b,h0)[0],
    'k': lambda x: wkv7_fwd(r,w,x,v,a,b,h0)[0],
    'v': lambda x: wkv7_fwd(r,w,k,x,a,b,h0)[0],
    'a': lambda x: wkv7_fwd(r,w,k,v,x,b,h0)[0],
    'b': lambda x: wkv7_fwd(r,w,k,v,a,x,h0)[0],
}
params = {'r':r,'k':k,'v':v,'a':a,'b':b}
grads  = {'r':dr,'k':dk,'v':dv,'a':da,'b':db}

print(f"{'param':<6} {'max_diff':>12} {'status':>6}")
print("-" * 28)
for name in ['r','k','v','a','b']:
    fn = lambda x, n=name: (fns[n](x) * d_out).sum()
    ng = num_grad(fn, params[name])
    diff = mx.max(mx.abs(grads[name] - ng)).item()
    print(f"{name:<6} {diff:>12.6f} {'OK' if diff < 1e-3 else 'FAIL':>6}")
