"""
wkv7_simd.py — оптимизированный backward через simd_sum
=========================================================
D=64 = 2 × simd_size(32) → simd_sum заменяет D×D shared accum.
2 барьера на timestep вместо 12. Shared mem: 3840 байт вместо ~18 KB.
"""
import mlx.core as mx

HEAD_SIZE = 64
CHUNK     = 32

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

_fwd_cache = {}

def _get_fwd(H):
    if H in _fwd_cache: return _fwd_cache[H]
    hdr = f"""
constant uint HEAD_SIZE_C = {HEAD_SIZE};
constant uint CHUNK_C     = {CHUNK};
constant uint H_C         = {H};
"""
    body = r"""
    uint dv  = thread_position_in_grid.y;
    uint bhi = thread_position_in_grid.x;
    uint bi  = bhi / H_C; uint hi = bhi % H_C;
    float h_row[HEAD_SIZE_C];
    uint h_base = (bi*H_C+hi)*HEAD_SIZE_C*HEAD_SIZE_C + dv*HEAD_SIZE_C;
    for (uint dk=0; dk<HEAD_SIZE_C; dk++) h_row[dk] = h_in[h_base+dk];
    for (uint t=0; t<CHUNK_C; t++) {
        uint base = ((bi*CHUNK_C+t)*H_C+hi)*HEAD_SIZE_C;
        float sa = 0.0f;
        for (uint dk=0; dk<HEAD_SIZE_C; dk++) sa += h_row[dk]*a[base+dk];
        sa_out[base+dv] = sa;
        float v_dv = v[base+dv];
        for (uint dk=0; dk<HEAD_SIZE_C; dk++)
            h_row[dk] = w[base+dk]*h_row[dk] + v_dv*k[base+dk] + sa*b[base+dk];
        float y = 0.0f;
        for (uint dk=0; dk<HEAD_SIZE_C; dk++) y += h_row[dk]*r[base+dk];
        out[base+dv] = y;
    }
    for (uint dk=0; dk<HEAD_SIZE_C; dk++) h_out[h_base+dk] = h_row[dk];
"""
    kern = mx.fast.metal_kernel(
        name=f"wkv7_fwd_simd_{H}",
        input_names=["r","w","k","v","a","b","h_in"],
        output_names=["out","h_out","sa_out"],
        header=hdr, source=body,
    )
    _fwd_cache[H] = kern
    return kern

_bwd_cache = {}

def _get_bwd_simd(H):
    if H in _bwd_cache: return _bwd_cache[H]
    assert HEAD_SIZE == 64, "Ядро рассчитано на D=64 (2 SIMD-группы по 32)"
    hdr = f"""
constant uint HEAD_SIZE_C = {HEAD_SIZE};
constant uint CHUNK_C     = {CHUNK};
constant uint H_C         = {H};
"""
    body = r"""
    uint dv        = thread_position_in_threadgroup.x;
    uint bhi       = threadgroup_position_in_grid.x;
    uint bi        = bhi / H_C;  uint hi = bhi % H_C;
    uint simd_id   = dv / 32;
    uint simd_lane = dv % 32;

    // Cross-thread shared: 5×D×4 = 1280 байт
    threadgroup float k_sh[HEAD_SIZE_C];
    threadgroup float r_sh[HEAD_SIZE_C];
    threadgroup float w_sh[HEAD_SIZE_C];
    threadgroup float a_sh[HEAD_SIZE_C];
    threadgroup float b_sh[HEAD_SIZE_C];
    // Буферы simd_sum: 5×2×D×4 = 2560 байт  (итого 3840 vs 18KB у v2)
    threadgroup float p0[2][HEAD_SIZE_C];  // dr
    threadgroup float p1[2][HEAD_SIZE_C];  // dw
    threadgroup float p2[2][HEAD_SIZE_C];  // da
    threadgroup float p3[2][HEAD_SIZE_C];  // dk
    threadgroup float p4[2][HEAD_SIZE_C];  // db

    float C_row[HEAD_SIZE_C];
    float h_row[HEAD_SIZE_C];
    uint h_base = (bi*H_C+hi)*HEAD_SIZE_C*HEAD_SIZE_C + dv*HEAD_SIZE_C;
    for (uint dk=0; dk<HEAD_SIZE_C; dk++) {
        C_row[dk] = d_h_out[h_base+dk];
        h_row[dk] = h_out_fwd[h_base+dk];
    }

    for (int t=(int)CHUNK_C-1; t>=0; t--) {
        uint base = ((bi*CHUNK_C+(uint)t)*H_C+hi)*HEAD_SIZE_C;

        // BARRIER 1: загрузка cross-thread векторов
        k_sh[dv] = k[base+dv];
        r_sh[dv] = r[base+dv];
        w_sh[dv] = w[base+dv];
        a_sh[dv] = a[base+dv];
        b_sh[dv] = b[base+dv];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float dy_dv = d_out[base+dv];
        float sa_dv = sa_fwd_in[base+dv];
        float v_dv  = v[base+dv];

        // C[dv,dk] += dy[dv]*r[dk]
        for (uint dk=0; dk<HEAD_SIZE_C; dk++)
            C_row[dk] += dy_dv * r_sh[dk];

        // dv_out и dsa: локально
        float dsa_dv=0, dv_val=0;
        for (uint dk=0; dk<HEAD_SIZE_C; dk++) {
            dsa_dv += C_row[dk] * b_sh[dk];
            dv_val  += C_row[dk] * k_sh[dk];
        }
        dv_out[base+dv] = dv_val;

        // Один цикл: simd_sum для всех 5 выходов + обновление h_row
        for (uint dk=0; dk<HEAD_SIZE_C; dk++) {
            float hp = (h_row[dk] - v_dv*k_sh[dk] - sa_dv*b_sh[dk]) / w_sh[dk];

            float ps0 = simd_sum(dy_dv     * h_row[dk]);
            float ps1 = simd_sum(C_row[dk] * hp);
            float ps2 = simd_sum(dsa_dv    * hp);
            float ps3 = simd_sum(C_row[dk] * v_dv);
            float ps4 = simd_sum(sa_dv     * C_row[dk]);

            if (simd_lane == 0) {
                p0[simd_id][dk] = ps0;
                p1[simd_id][dk] = ps1;
                p2[simd_id][dk] = ps2;
                p3[simd_id][dk] = ps3;
                p4[simd_id][dk] = ps4;
            }
            h_row[dk] = hp;
        }

        // C_row = C*w + dsa*a  (w_sh, a_sh валидны — загружены в barrier1)
        for (uint dk=0; dk<HEAD_SIZE_C; dk++)
            C_row[dk] = C_row[dk]*w_sh[dk] + dsa_dv*a_sh[dk];

        // BARRIER 2: синхронизация p0..p4 + защита w_sh от перезаписи
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Сумма двух SIMD-групп: 2 сложения на выход
        dr_out[base+dv] = p0[0][dv] + p0[1][dv];
        dw_out[base+dv] = p1[0][dv] + p1[1][dv];
        da_out[base+dv] = p2[0][dv] + p2[1][dv];
        dk_out[base+dv] = p3[0][dv] + p3[1][dv];
        db_out[base+dv] = p4[0][dv] + p4[1][dv];
    }

    for (uint dk=0; dk<HEAD_SIZE_C; dk++) dh_in_out[h_base+dk] = C_row[dk];
"""
    kern = mx.fast.metal_kernel(
        name=f"wkv7_bwd_simd_H{H}",
        input_names=["r","w","k","v","a","b","h_out_fwd","sa_fwd_in","d_out","d_h_out"],
        output_names=["dr_out","dw_out","dk_out","dv_out","da_out","db_out","dh_in_out"],
        header=hdr, source=body,
        atomic_outputs=False,
    )
    _bwd_cache[H] = kern
    return kern

@mx.custom_function
def _wkv7_chunk_simd(r, w, k, v, a, b, h_in):
    B, T, H, D = r.shape
    res = _get_fwd(H)(
        inputs=[x.astype(mx.float32) for x in [r,w,k,v,a,b,h_in]],
        grid=(B*H, D, 1), threadgroup=(1, 1, 1),
        output_shapes=[(B,T,H,D), (B,H,D,D), (B,T,H,D)],
        output_dtypes=[mx.float32]*3,
    )
    return res[0], res[1], res[2]

@_wkv7_chunk_simd.vjp
def _wkv7_chunk_simd_vjp(primals, cotangents, outputs):
    r, w, k, v, a, b, h_in = primals
    d_out, d_h_out, _       = cotangents
    _, h_out_fwd, sa_fwd    = outputs
    mx.eval(h_out_fwd, sa_fwd, d_out, d_h_out)
    B, T, H, D = r.shape
    res = _get_bwd_simd(H)(
        inputs=[x.astype(mx.float32) for x in
                [r,w,k,v,a,b,h_out_fwd,sa_fwd,d_out,d_h_out]],
        grid=(B*H*D, 1, 1), threadgroup=(D, 1, 1),
        output_shapes=[(B,T,H,D)]*6 + [(B,H,D,D)],
        output_dtypes=[mx.float32]*7,
    )
    return res[0], res[1], res[2], res[3], res[4], res[5], res[6]

def wkv7_simd_train(r, w, k, v, a, b):
    B, T, H, D = r.shape
    h = mx.zeros((B, H, D, D))
    outs = []
    for start in range(0, T, CHUNK):
        end = min(start + CHUNK, T); cl = end - start
        rc,wc,kc,vc,ac,bc = (x[:,start:end] for x in (r,w,k,v,a,b))
        if cl < CHUNK:
            pad = CHUNK - cl
            def p(x, val=0.0):
                return mx.pad(x,[(0,0),(0,pad),(0,0),(0,0)],constant_values=val)
            rc=p(rc); wc=p(wc,1.0); kc=p(kc); vc=p(vc); ac=p(ac); bc=p(bc)
        out_c, h, _ = _wkv7_chunk_simd(rc,wc,kc,vc,ac,bc,h)
        mx.eval(h, out_c)
        outs.append(out_c[:,:cl])
    return mx.concatenate(outs, axis=1)
