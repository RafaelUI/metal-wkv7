import mlx.core as mx

HEAD_SIZE = 64
CHUNK     = 32

# ─── Python forward (для recompute в VJP) ───────────────────────────────────

def _py_fwd_chunk(r, w, k, v, a, b, h):
    B, T, H, D = r.shape
    outs, h_all, sa_all = [], [h], []
    for t in range(T):
        r_t,w_t,k_t,v_t,a_t,b_t = r[:,t],w[:,t],k[:,t],v[:,t],a[:,t],b[:,t]
        sa  = mx.einsum("bhsd,bhd->bhs", h, a_t)
        h   = h * w_t[:,:,None,:] + mx.einsum("bhs,bhd->bhsd",v_t,k_t) + mx.einsum("bhs,bhd->bhsd",sa,b_t)
        outs.append(mx.einsum("bhsd,bhd->bhs", h, r_t))
        h_all.append(h); sa_all.append(sa)
    return mx.stack(outs, axis=1), h, h_all, sa_all

def _py_bwd_chunk(r, w, k, v, a, b, h_all, sa_all, d_out, d_h):
    B, T, H, D = r.shape
    C = d_h
    dr_l, dw_l, dk_l, dv_l, da_l, db_l = [], [], [], [], [], []
    for t in range(T-1, -1, -1):
        r_t,w_t,k_t,v_t,a_t,b_t = r[:,t],w[:,t],k[:,t],v[:,t],a[:,t],b[:,t]
        dy_t, h_prev, h_cur, sa_t = d_out[:,t], h_all[t], h_all[t+1], sa_all[t]
        C   = C + mx.einsum("bhs,bhd->bhsd", dy_t, r_t)
        dsa = mx.einsum("bhsd,bhd->bhs", C, b_t)
        dr_l.insert(0, mx.einsum("bhs,bhsd->bhd", dy_t, h_cur))
        dw_l.insert(0, mx.sum(C * h_prev, axis=2))
        dk_l.insert(0, mx.einsum("bhsd,bhs->bhd", C, v_t))
        dv_l.insert(0, mx.einsum("bhsd,bhd->bhs", C, k_t))
        db_l.insert(0, mx.einsum("bhs,bhsd->bhd", sa_t, C))
        da_l.insert(0, mx.einsum("bhs,bhsd->bhd", dsa, h_prev))
        C = C * w_t[:,:,None,:] + mx.einsum("bhs,bhd->bhsd", dsa, a_t)
    return (mx.stack(dr_l,1), mx.stack(dw_l,1), mx.stack(dk_l,1),
            mx.stack(dv_l,1), mx.stack(da_l,1), mx.stack(db_l,1), C)

# ─── Metal forward kernel ────────────────────────────────────────────────────

_metal_cache = {}

def _get_metal_fwd(H: int):
    if H in _metal_cache:
        return _metal_cache[H]
    header = f"""
constant uint HEAD_SIZE_C = {HEAD_SIZE};
constant uint CHUNK_C     = {CHUNK};
constant uint H_C         = {H};
"""
    source = """
    uint dv  = thread_position_in_grid.y;
    uint bhi = thread_position_in_grid.x;
    uint bi  = bhi / H_C;
    uint hi  = bhi % H_C;

    float h_row[HEAD_SIZE_C];
    uint h_base = (bi * H_C + hi) * HEAD_SIZE_C * HEAD_SIZE_C + dv * HEAD_SIZE_C;
    for (uint dk = 0; dk < HEAD_SIZE_C; dk++) {
        h_row[dk] = h_in[h_base + dk];
    }

    for (uint t = 0; t < CHUNK_C; t++) {
        uint base = ((bi * CHUNK_C + t) * H_C + hi) * HEAD_SIZE_C;

        float sa = 0.0f;
        for (uint dk = 0; dk < HEAD_SIZE_C; dk++) {
            sa += h_row[dk] * a[base + dk];
        }
        float v_dv = v[base + dv];
        for (uint dk = 0; dk < HEAD_SIZE_C; dk++) {
            h_row[dk] = w[base + dk] * h_row[dk]
                      + v_dv * k[base + dk]
                      + sa   * b[base + dk];
        }
        float y = 0.0f;
        for (uint dk = 0; dk < HEAD_SIZE_C; dk++) {
            y += h_row[dk] * r[base + dk];
        }
        out[base + dv] = y;
    }
    for (uint dk = 0; dk < HEAD_SIZE_C; dk++) {
        h_out[h_base + dk] = h_row[dk];
    }
"""
    kernel = mx.fast.metal_kernel(
        name=f"wkv7_fwd_H{H}",
        input_names=["r","w","k","v","a","b","h_in"],
        output_names=["out","h_out"],
        header=header, source=source,
    )
    _metal_cache[H] = kernel
    return kernel

# ─── custom_function: Metal forward + Python VJP ────────────────────────────

@mx.custom_function
def wkv7_chunk(r, w, k, v, a, b, h_in):
    B, T, H, D = r.shape
    assert D == HEAD_SIZE and T == CHUNK
    kernel = _get_metal_fwd(H)
    result = kernel(
        inputs=[r.astype(mx.float32), w.astype(mx.float32),
                k.astype(mx.float32), v.astype(mx.float32),
                a.astype(mx.float32), b.astype(mx.float32),
                h_in.astype(mx.float32)],
        grid=(B * H, D, 1), threadgroup=(1, 1, 1),
        output_shapes=[(B, T, H, D), (B, H, D, D)],
        output_dtypes=[mx.float32, mx.float32],
    )
    return result[0], result[1]

@wkv7_chunk.vjp
def wkv7_chunk_vjp(primals, cotangents, outputs):
    r, w, k, v, a, b, h_in = primals
    d_out, d_h_out = cotangents
    # Recompute intermediate states via Python forward
    _, _, h_all, sa_all = _py_fwd_chunk(r, w, k, v, a, b, h_in)
    # Python backward
    dr, dw, dk, dv, da, db, d_h_in = _py_bwd_chunk(
        r, w, k, v, a, b, h_all, sa_all, d_out, d_h_out
    )
    return dr, dw, dk, dv, da, db, d_h_in

# ─── Публичный API ──────────────────────────────────────────────────────────

def wkv7_fast(r, w, k, v, a, b):
    B, T, H, D = r.shape
    h    = mx.zeros((B, H, D, D))
    outs = []
    for start in range(0, T, CHUNK):
        end = min(start + CHUNK, T)
        cl  = end - start
        rc,wc,kc,vc,ac,bc = (x[:,start:end] for x in (r,w,k,v,a,b))
        if cl < CHUNK:
            pad = CHUNK - cl
            def p(x, val=0.0):
                return mx.pad(x,[(0,0),(0,pad),(0,0),(0,0)],constant_values=val)
            rc=p(rc); wc=p(wc,1.0); kc=p(kc); vc=p(vc); ac=p(ac); bc=p(bc)
        out_c, h = wkv7_chunk(rc, wc, kc, vc, ac, bc, h)
        outs.append(out_c[:, :cl])
    return mx.concatenate(outs, axis=1)
