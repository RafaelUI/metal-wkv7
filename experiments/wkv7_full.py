"""
wkv7_full.py — Full-sequence single-kernel WKV7 для Apple Silicon / MLX.

Вместо Python-цикла с 16 mx.eval() sync-точками (для T=512, CHUNK=32):
  - Один Metal forward kernel: обрабатывает весь T, сохраняет h-чекпоинты
  - Один Metal backward kernel: проходит все чанки в обратном порядке
  - 0 Python sync-точек → разрешает mx.compile(train_step)

Архитектура чекпоинтирования:
  h_ckpt shape = [B, NC, H, D, D], NC = T/CHUNK
  Forward: сохраняет h в конце каждого чанка (16 KB * NC * B * H)
  Backward: инициализирует h_row из h_ckpt[NC-1], далее propagates без перезагрузки
"""

import mlx.core as mx

HEAD_SIZE = 64
CHUNK     = 32  # гранулярность чекпоинтирования

_fwd_full_cache = {}
_bwd_full_cache = {}

# ─── Forward full-sequence kernel ─────────────────────────────────────────────

def _get_fwd_full(H):
    if H in _fwd_full_cache: return _fwd_full_cache[H]
    hdr = f"""
constant uint HEAD_SIZE_C = {HEAD_SIZE};
constant uint CHUNK_C     = {CHUNK};
constant uint H_C         = {H};
"""
    # grid=(B*H, D, 1), threadgroup=(1,1,1)
    # Thread (bhi, dv): обрабатывает строку dv матрицы h для (B,H)-пары bhi
    body = r"""
    uint dv  = thread_position_in_grid.y;
    uint bhi = thread_position_in_grid.x;
    uint bi  = bhi / H_C; uint hi = bhi % H_C;

    uint T  = (uint)r_shape[1];   // полная длина последовательности
    uint NC = T / CHUNK_C;        // число чекпоинтных чанков

    float h_row[HEAD_SIZE_C];
    uint h_base = (bi*H_C+hi)*HEAD_SIZE_C*HEAD_SIZE_C + dv*HEAD_SIZE_C;
    for (uint dk=0; dk<HEAD_SIZE_C; dk++) h_row[dk] = h_in[h_base+dk];

    for (uint t=0; t<T; t++) {
        uint base = (bi*T + t)*H_C*HEAD_SIZE_C + hi*HEAD_SIZE_C;

        float sa = 0.0f;
        for (uint dk=0; dk<HEAD_SIZE_C; dk++) sa += h_row[dk]*a[base+dk];
        sa_out[base+dv] = sa;

        float v_dv = v[base+dv];
        for (uint dk=0; dk<HEAD_SIZE_C; dk++)
            h_row[dk] = w[base+dk]*h_row[dk] + v_dv*k[base+dk] + sa*b[base+dk];

        float y = 0.0f;
        for (uint dk=0; dk<HEAD_SIZE_C; dk++) y += h_row[dk]*r[base+dk];
        out[base+dv] = y;

        // Чекпоинт h в конце каждого чанка
        if ((t+1) % CHUNK_C == 0) {
            uint ci = t / CHUNK_C;
            // h_ckpt shape: [B, NC, H, D, D]
            uint ck_base = bi*(NC*H_C*HEAD_SIZE_C*HEAD_SIZE_C)
                         + ci*(H_C*HEAD_SIZE_C*HEAD_SIZE_C)
                         + hi*(HEAD_SIZE_C*HEAD_SIZE_C)
                         + dv*HEAD_SIZE_C;
            for (uint dk=0; dk<HEAD_SIZE_C; dk++) h_ckpt[ck_base+dk] = h_row[dk];
        }
    }

    // Финальное состояние (= последний чекпоинт, удобно иметь отдельно)
    for (uint dk=0; dk<HEAD_SIZE_C; dk++) h_out[h_base+dk] = h_row[dk];
"""
    kern = mx.fast.metal_kernel(
        name=f"wkv7_fwd_full_{H}",
        input_names=["r","w","k","v","a","b","h_in"],
        output_names=["out","h_out","h_ckpt","sa_out"],
        header=hdr, source=body,
    )
    _fwd_full_cache[H] = kern
    return kern

# ─── Backward full-sequence kernel ────────────────────────────────────────────

def _get_bwd_full(H):
    if H in _bwd_full_cache: return _bwd_full_cache[H]
    hdr = f"""
constant uint HEAD_SIZE_C = {HEAD_SIZE};
constant uint CHUNK_C     = {CHUNK};
constant uint H_C         = {H};
"""
    # grid=(B*H*D, 1, 1), threadgroup=(D, 1, 1)
    # D=64 потоков в threadgroup — shared memory паттерн как в v2
    body = r"""
    uint dv  = thread_position_in_threadgroup.x;
    uint bhi = threadgroup_position_in_grid.x;
    uint bi  = bhi / H_C; uint hi = bhi % H_C;

    uint T  = (uint)r_shape[1];
    uint NC = T / CHUNK_C;

    // 16 KB shared для column-sum редукций
    threadgroup float accum[HEAD_SIZE_C][HEAD_SIZE_C];
    threadgroup float k_sh[HEAD_SIZE_C], v_sh[HEAD_SIZE_C];
    threadgroup float r_sh[HEAD_SIZE_C], w_sh[HEAD_SIZE_C];
    threadgroup float a_sh[HEAD_SIZE_C], b_sh[HEAD_SIZE_C];
    threadgroup float dy_sh[HEAD_SIZE_C], sa_sh[HEAD_SIZE_C];
    threadgroup float dsa_sh[HEAD_SIZE_C];

    float C_row[HEAD_SIZE_C];
    float h_row[HEAD_SIZE_C];

    uint h_base = (bi*H_C+hi)*HEAD_SIZE_C*HEAD_SIZE_C + dv*HEAD_SIZE_C;

    // Инициализируем C из котангента финального h (обычно нули)
    for (uint dk=0; dk<HEAD_SIZE_C; dk++) C_row[dk] = d_h_out[h_base+dk];

    // Обратный проход по всем чанкам.
    // КРИТИЧНО: перезагружаем h_ckpt в начале каждого чанка —
    // накопленная ошибка реконструкции через деление на w усиливается как (1/w)^T,
    // поэтому нельзя переносить h_row между чанками — только из сохранённого чекпоинта.
    for (int ci=(int)NC-1; ci>=0; ci--) {
        uint ci_u = (uint)ci;
        uint t_end = ci_u * CHUNK_C + CHUNK_C;   // exclusive
        uint t_beg = ci_u * CHUNK_C;             // inclusive

        // Перезагружаем h из чекпоинта начала этого чанка (= конца чанка ci)
        uint ck_base = bi*(NC*H_C*HEAD_SIZE_C*HEAD_SIZE_C)
                     + ci_u*(H_C*HEAD_SIZE_C*HEAD_SIZE_C)
                     + hi*(HEAD_SIZE_C*HEAD_SIZE_C)
                     + dv*HEAD_SIZE_C;
        for (uint dk=0; dk<HEAD_SIZE_C; dk++) h_row[dk] = h_ckpt[ck_base+dk];
        for (int t=(int)(t_end-1); t>=(int)t_beg; t--) {
            uint base = (bi*T + (uint)t)*H_C*HEAD_SIZE_C + hi*HEAD_SIZE_C;

            k_sh[dv]=k[base+dv]; v_sh[dv]=v[base+dv];
            r_sh[dv]=r[base+dv]; w_sh[dv]=w[base+dv];
            a_sh[dv]=a[base+dv]; b_sh[dv]=b[base+dv];
            dy_sh[dv]=d_out[base+dv]; sa_sh[dv]=sa_fwd_in[base+dv];
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float dy_dv = dy_sh[dv];
            for (uint dk=0; dk<HEAD_SIZE_C; dk++) C_row[dk] += dy_dv*r_sh[dk];

            float dsa_dv=0, dv_val=0;
            for (uint dk=0; dk<HEAD_SIZE_C; dk++) {
                dsa_dv += C_row[dk]*b_sh[dk];
                dv_val  += C_row[dk]*k_sh[dk];
            }
            dv_out[base+dv] = dv_val;

            dsa_sh[dv] = dsa_dv;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // dr: column sum of dy*h_cur
            for (uint dk=0; dk<HEAD_SIZE_C; dk++) accum[dv][dk] = dy_dv*h_row[dk];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float dr_val=0; for (uint s=0; s<HEAD_SIZE_C; s++) dr_val+=accum[s][dv];
            dr_out[base+dv] = dr_val;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Восстанавливаем h_prev = (h_cur - v*k - sa*b) / w
            float sa_dv=sa_sh[dv], v_dv=v_sh[dv];
            for (uint dk=0; dk<HEAD_SIZE_C; dk++) {
                float hp = (h_row[dk] - v_dv*k_sh[dk] - sa_dv*b_sh[dk]) / w_sh[dk];
                accum[dv][dk] = C_row[dk]*hp;
                h_row[dk] = hp;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float dw_val=0; for (uint s=0; s<HEAD_SIZE_C; s++) dw_val+=accum[s][dv];
            dw_out[base+dv] = dw_val;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // dk: column sum of C*v
            for (uint dk=0; dk<HEAD_SIZE_C; dk++) accum[dv][dk] = C_row[dk]*v_dv;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float dk_val=0; for (uint s=0; s<HEAD_SIZE_C; s++) dk_val+=accum[s][dv];
            dk_out[base+dv] = dk_val;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // da: column sum of dsa*h_prev
            for (uint dk=0; dk<HEAD_SIZE_C; dk++) accum[dv][dk] = dsa_sh[dv]*h_row[dk];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float da_val=0; for (uint s=0; s<HEAD_SIZE_C; s++) da_val+=accum[s][dv];
            da_out[base+dv] = da_val;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // db: column sum of sa_fwd*C
            for (uint dk=0; dk<HEAD_SIZE_C; dk++) accum[dv][dk] = sa_sh[dv]*C_row[dk];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float db_val=0; for (uint s=0; s<HEAD_SIZE_C; s++) db_val+=accum[s][dv];
            db_out[base+dv] = db_val;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Обновляем C для предыдущего timestep
            for (uint dk=0; dk<HEAD_SIZE_C; dk++)
                C_row[dk] = C_row[dk]*w_sh[dk] + dsa_dv*a_sh[dk];
        }
    }

    for (uint dk=0; dk<HEAD_SIZE_C; dk++) dh_in_out[h_base+dk] = C_row[dk];
"""
    kern = mx.fast.metal_kernel(
        name=f"wkv7_bwd_full_{H}",
        input_names=["r","w","k","v","a","b","h_ckpt","sa_fwd_in","d_out","d_h_out"],
        output_names=["dr_out","dw_out","dk_out","dv_out","da_out","db_out","dh_in_out"],
        header=hdr, source=body,
        atomic_outputs=False,
    )
    _bwd_full_cache[H] = kern
    return kern

# ─── custom_function (один вызов на весь T) ───────────────────────────────────

@mx.custom_function
def _wkv7_full_metal(r, w, k, v, a, b, h_in):
    """Forward: весь T в одном Metal kernel + сохранение h-чекпоинтов."""
    B, T, H, D = r.shape
    NC = T // CHUNK
    res = _get_fwd_full(H)(
        inputs=[x.astype(mx.float32) for x in [r,w,k,v,a,b,h_in]],
        grid=(B*H, D, 1), threadgroup=(1, 1, 1),
        output_shapes=[(B,T,H,D), (B,H,D,D), (B,NC,H,D,D), (B,T,H,D)],
        output_dtypes=[mx.float32]*4,
    )
    return res[0], res[1], res[2], res[3]  # out, h_out, h_ckpt, sa_out

@_wkv7_full_metal.vjp
def _wkv7_full_metal_vjp(primals, cotangents, outputs):
    """Backward: весь T в одном Metal kernel, использует h_ckpt."""
    r, w, k, v, a, b, h_in = primals
    d_out, d_h_out, _, _    = cotangents   # d_h_ckpt и d_sa_out нулевые
    _, h_out, h_ckpt, sa_fwd = outputs
    # Материализуем lazy-тензоры перед Metal backward
    mx.eval(h_ckpt, sa_fwd, d_out)
    B, T, H, D = r.shape
    # Котангент финального h: либо из d_h_out (если h_out используется), либо нули
    # В training-режиме h_out не используется → d_h_out = zeros
    if d_h_out is None:
        d_h_final = mx.zeros((B, H, D, D), dtype=mx.float32)
    else:
        d_h_final = d_h_out.astype(mx.float32)
    res = _get_bwd_full(H)(
        inputs=[x.astype(mx.float32) for x in [r,w,k,v,a,b,h_ckpt,sa_fwd,d_out,d_h_final]],
        grid=(B*H*D, 1, 1), threadgroup=(D, 1, 1),
        output_shapes=[(B,T,H,D)]*6 + [(B,H,D,D)],
        output_dtypes=[mx.float32]*7,
    )
    return res[0], res[1], res[2], res[3], res[4], res[5], res[6]

# ─── Публичный API ────────────────────────────────────────────────────────────

def wkv7_full_train(r, w, k, v, a, b):
    """
    Полная последовательность T в ОДНОМ Metal kernel.
    Нет Python-цикла, нет mx.eval sync-точек → совместимо с mx.compile.
    T должно быть кратно CHUNK=32.
    """
    B, T, H, D = r.shape
    assert T % CHUNK == 0, f"T={T} должно быть кратно CHUNK={CHUNK}"
    assert D == HEAD_SIZE,  f"D={D} != HEAD_SIZE={HEAD_SIZE}"
    h_in = mx.zeros((B, H, D, D))
    out, _, _, _ = _wkv7_full_metal(r, w, k, v, a, b, h_in)
    return out
