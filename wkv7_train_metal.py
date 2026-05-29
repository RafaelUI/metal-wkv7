import mlx.core as mx

HEAD_SIZE = 64
CHUNK     = 32

_fwd_cache = {}
_bwd_cache = {}

# ─── Forward kernel (сохраняет sa для backward) ──────────────────────────────

def _get_fwd(H):
    if H in _fwd_cache: return _fwd_cache[H]
    hdr = f"""
constant uint HEAD_SIZE_C = {HEAD_SIZE};
constant uint CHUNK_C     = {CHUNK};
constant uint H_C         = {H};
"""
    # threadgroup=(1,1,1) — нет shared, читаем прямо из global
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
    k = mx.fast.metal_kernel(
        name=f"wkv7_fwd_{H}",
        input_names=["r","w","k","v","a","b","h_in"],
        output_names=["out","h_out","sa_out"],
        header=hdr, source=body,
    )
    _fwd_cache[H] = k
    return k

# ─── Backward kernel (device atomics через atomic_outputs=True) ───────────────

def _get_bwd(H):
    if H in _bwd_cache: return _bwd_cache[H]
    hdr = f"""
constant uint HEAD_SIZE_C = {HEAD_SIZE};
constant uint CHUNK_C     = {CHUNK};
constant uint H_C         = {H};
"""
    # Без атомиков — threadgroup shared memory редукция
    # Grid: (B*H, 1, 1), threadgroup: (D, 1, 1)
    # Thread dv=i, D потоков в одном threadgroup
    # accum[D][D] = 16KB — переиспользуется для каждого градиента
    body = r"""
    uint dv  = thread_position_in_threadgroup.x;
    uint bhi = threadgroup_position_in_grid.x;
    uint bi  = bhi / H_C; uint hi = bhi % H_C;

    // 16KB shared для редукций (переиспользуется)
    threadgroup float accum[HEAD_SIZE_C][HEAD_SIZE_C];
    // Входные векторы в shared
    threadgroup float k_sh[HEAD_SIZE_C], v_sh[HEAD_SIZE_C];
    threadgroup float r_sh[HEAD_SIZE_C], w_sh[HEAD_SIZE_C];
    threadgroup float a_sh[HEAD_SIZE_C], b_sh[HEAD_SIZE_C];
    threadgroup float dy_sh[HEAD_SIZE_C], sa_sh[HEAD_SIZE_C];
    threadgroup float dsa_sh[HEAD_SIZE_C];

    float C_row[HEAD_SIZE_C];
    float h_row[HEAD_SIZE_C];

    uint h_base = (bi*H_C+hi)*HEAD_SIZE_C*HEAD_SIZE_C + dv*HEAD_SIZE_C;
    for (uint dk=0; dk<HEAD_SIZE_C; dk++) {
        C_row[dk] = d_h_out[h_base+dk];
        h_row[dk] = h_out_fwd[h_base+dk];
    }

    for (int t=(int)CHUNK_C-1; t>=0; t--) {
        uint base = ((bi*CHUNK_C+(uint)t)*H_C+hi)*HEAD_SIZE_C;

        // Загружаем векторы в shared
        k_sh[dv]=k[base+dv]; v_sh[dv]=v[base+dv];
        r_sh[dv]=r[base+dv]; w_sh[dv]=w[base+dv];
        a_sh[dv]=a[base+dv]; b_sh[dv]=b[base+dv];
        dy_sh[dv]=d_out[base+dv]; sa_sh[dv]=sa_fwd_in[base+dv];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // C[dv,dk] += dy[dv]*r[dk]
        float dy_dv = dy_sh[dv];
        for (uint dk=0; dk<HEAD_SIZE_C; dk++) C_row[dk] += dy_dv*r_sh[dk];

        // dsa[dv], dv_out[dv] — локально
        float dsa_dv=0, dv_val=0;
        for (uint dk=0; dk<HEAD_SIZE_C; dk++) {
            dsa_dv += C_row[dk]*b_sh[dk];
            dv_val  += C_row[dk]*k_sh[dk];
        }
        dv_out[base+dv] = dv_val;

        // Сохраняем dsa в shared
        dsa_sh[dv] = dsa_dv;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Фаза dr: accum[dv][dk] = dy[dv]*h_cur[dv,dk]
        for (uint dk=0; dk<HEAD_SIZE_C; dk++) accum[dv][dk] = dy_dv*h_row[dk];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float dr_val=0; for (uint s=0; s<HEAD_SIZE_C; s++) dr_val+=accum[s][dv];
        dr_out[base+dv] = dr_val;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Фаза dw: h_prev[dv,dk] = (h_cur - v*k - sa*b)/w
        // accum[dv][dk] = C[dv,dk]*h_prev[dv,dk], обновляем h_row
        float sa_dv=sa_sh[dv], v_dv=v_sh[dv];
        for (uint dk=0; dk<HEAD_SIZE_C; dk++) {
            float hp = (h_row[dk] - v_dv*k_sh[dk] - sa_dv*b_sh[dk]) / w_sh[dk];
            accum[dv][dk] = C_row[dk]*hp;
            h_row[dk] = hp;  // обновляем для следующего шага
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float dw_val=0; for (uint s=0; s<HEAD_SIZE_C; s++) dw_val+=accum[s][dv];
        dw_out[base+dv] = dw_val;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Фаза dk_grad: accum[dv][dk] = C[dv,dk]*v[dv]
        for (uint dk=0; dk<HEAD_SIZE_C; dk++) accum[dv][dk] = C_row[dk]*v_dv;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float dk_val=0; for (uint s=0; s<HEAD_SIZE_C; s++) dk_val+=accum[s][dv];
        dk_out[base+dv] = dk_val;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Фаза da: accum[dv][dk] = dsa[dv]*h_prev[dv,dk]  (h_row = h_prev)
        for (uint dk=0; dk<HEAD_SIZE_C; dk++) accum[dv][dk] = dsa_sh[dv]*h_row[dk];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float da_val=0; for (uint s=0; s<HEAD_SIZE_C; s++) da_val+=accum[s][dv];
        da_out[base+dv] = da_val;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Фаза db: accum[dv][dk] = sa_fwd[dv]*C[dv,dk]
        for (uint dk=0; dk<HEAD_SIZE_C; dk++) accum[dv][dk] = sa_sh[dv]*C_row[dk];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float db_val=0; for (uint s=0; s<HEAD_SIZE_C; s++) db_val+=accum[s][dv];
        db_out[base+dv] = db_val;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Обновляем C
        for (uint dk=0; dk<HEAD_SIZE_C; dk++)
            C_row[dk] = C_row[dk]*w_sh[dk] + dsa_dv*a_sh[dk];
    }

    // Котангент начального состояния
    for (uint dk=0; dk<HEAD_SIZE_C; dk++) dh_in_out[h_base+dk] = C_row[dk];
"""
    k = mx.fast.metal_kernel(
        name           = f"wkv7_bwd_v2_H{H}",
        input_names    = ["r","w","k","v","a","b","h_out_fwd","sa_fwd_in","d_out","d_h_out"],
        output_names   = ["dr_out","dw_out","dk_out","dv_out","da_out","db_out","dh_in_out"],
        header         = hdr,
        source         = body,
        atomic_outputs = False,  # Без атомиков!
    )
    _bwd_cache[H] = k
    return k

# ─── custom_function ─────────────────────────────────────────────────────────

@mx.custom_function
def wkv7_chunk(r, w, k, v, a, b, h_in):
    B, T, H, D = r.shape
    res = _get_fwd(H)(
        inputs=[x.astype(mx.float32) for x in [r,w,k,v,a,b,h_in]],
        grid=(B*H, D, 1), threadgroup=(1, 1, 1),
        output_shapes=[(B,T,H,D), (B,H,D,D), (B,T,H,D)],
        output_dtypes=[mx.float32]*3,
    )
    return res[0], res[1], res[2]

@wkv7_chunk.vjp
def wkv7_chunk_vjp(primals, cotangents, outputs):
    r, w, k, v, a, b, h_in = primals
    d_out, d_h_out, _       = cotangents
    _, h_out_fwd, sa_fwd    = outputs
    # Материализуем lazy массивы перед Metal kernel
    mx.eval(h_out_fwd, sa_fwd, d_out, d_h_out)
    B, T, H, D = r.shape
    res = _get_bwd(H)(
        inputs=[x.astype(mx.float32) for x in
                [r,w,k,v,a,b,h_out_fwd,sa_fwd,d_out,d_h_out]],
        grid=(B*H*D, 1, 1), threadgroup=(D, 1, 1),
        output_shapes=[(B,T,H,D)]*6 + [(B,H,D,D)],
        output_dtypes=[mx.float32]*7,
    )
    return res[0], res[1], res[2], res[3], res[4], res[5], res[6]

# ─── Публичный API ────────────────────────────────────────────────────────────

def wkv7_metal_train(r, w, k, v, a, b):
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
            rc=p(rc);wc=p(wc,1.0);kc=p(kc);vc=p(vc);ac=p(ac);bc=p(bc)
        out_c, h, _ = wkv7_chunk(rc,wc,kc,vc,ac,bc,h)
        outs.append(out_c[:,:cl])
    return mx.concatenate(outs, axis=1)
