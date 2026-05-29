"""
wkv7_bwd_v3 — два изменения относительно v2:

1. accum[D][D+1] вместо accum[D][D]
   64-way bank conflict → 2-way на запись И чтение.
   Запись: все D потоков пишут для фиксированного dk →
     old: bank = (dv*64+dk)%32 = dk%32 для всех dv → 64-way
     new: bank = (dv*65+dk)%32 = (dv+dk)%32 → все разные → 2-way

2. Убираем self-read shared (v_sh, dy_sh, sa_sh, dsa_sh) →
   минус одна threadgroup_barrier (была только для dsa_sh).
   k_sh, r_sh, w_sh, a_sh, b_sh остаются — они реально cross-thread.
"""
import mlx.core as mx

HEAD_SIZE = 64
CHUNK     = 32

_bwd_cache = {}

def _get_bwd_v3(H):
    if H in _bwd_cache: return _bwd_cache[H]
    hdr = f"""
constant uint HEAD_SIZE_C = {HEAD_SIZE};
constant uint CHUNK_C     = {CHUNK};
constant uint H_C         = {H};
"""
    body = r"""
    uint dv  = thread_position_in_threadgroup.x;
    uint bhi = threadgroup_position_in_grid.x;
    uint bi  = bhi / H_C;  uint hi = bhi % H_C;

    // ── Shared memory ────────────────────────────────────────────────────────
    // Cross-thread входные: 5×D×4 = 1280 байт
    threadgroup float k_sh[HEAD_SIZE_C];
    threadgroup float r_sh[HEAD_SIZE_C];
    threadgroup float w_sh[HEAD_SIZE_C];
    threadgroup float a_sh[HEAD_SIZE_C];
    threadgroup float b_sh[HEAD_SIZE_C];

    // accum[D][D+1] — +1 padding убирает 64-way → 2-way bank conflict
    // Размер: 64×65×4 = 16640 байт  (было 64×64×4 = 16384)
    // Банки при записи: bank=(dv*65+dk)%32=(dv+dk)%32 → все разные → нет конфликтов
    threadgroup float accum[HEAD_SIZE_C][HEAD_SIZE_C + 1];

    float C_row[HEAD_SIZE_C];
    float h_row[HEAD_SIZE_C];

    uint h_base = (bi*H_C+hi)*HEAD_SIZE_C*HEAD_SIZE_C + dv*HEAD_SIZE_C;
    for (uint dk=0; dk<HEAD_SIZE_C; dk++) {
        C_row[dk] = d_h_out[h_base+dk];
        h_row[dk] = h_out_fwd[h_base+dk];
    }

    for (int t=(int)CHUNK_C-1; t>=0; t--) {
        uint base = ((bi*CHUNK_C+(uint)t)*H_C+hi)*HEAD_SIZE_C;

        // ── BARRIER 1: загрузка cross-thread векторов ─────────────────────────
        k_sh[dv] = k[base+dv];
        r_sh[dv] = r[base+dv];
        w_sh[dv] = w[base+dv];
        a_sh[dv] = a[base+dv];
        b_sh[dv] = b[base+dv];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Локальные загрузки: читаем напрямую из global (self-read, не нужен shared)
        float dy_dv = d_out[base+dv];
        float sa_dv = sa_fwd_in[base+dv];
        float v_dv  = v[base+dv];

        // C[dv,dk] += dy[dv]*r[dk]
        for (uint dk=0; dk<HEAD_SIZE_C; dk++)
            C_row[dk] += dy_dv * r_sh[dk];

        // dsa[dv] и dv_out[dv] — локально (dot product по регистрам)
        float dsa_dv=0, dv_val=0;
        for (uint dk=0; dk<HEAD_SIZE_C; dk++) {
            dsa_dv += C_row[dk] * b_sh[dk];
            dv_val  += C_row[dk] * k_sh[dk];
        }
        dv_out[base+dv] = dv_val;
        // dsa_dv остаётся в регистре — нет нужды в shared memory

        // ── Фаза dr: dy[dv]*h_cur[dv,dk] ────────────────────────────────────
        for (uint dk=0; dk<HEAD_SIZE_C; dk++) accum[dv][dk] = dy_dv*h_row[dk];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float dr_val=0; for (uint s=0; s<HEAD_SIZE_C; s++) dr_val+=accum[s][dv];
        dr_out[base+dv] = dr_val;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── Фаза dw: C[dv,dk]*h_prev[dv,dk], реконструируем h_prev ──────────
        for (uint dk=0; dk<HEAD_SIZE_C; dk++) {
            float hp = (h_row[dk] - v_dv*k_sh[dk] - sa_dv*b_sh[dk]) / w_sh[dk];
            accum[dv][dk] = C_row[dk] * hp;
            h_row[dk] = hp;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float dw_val=0; for (uint s=0; s<HEAD_SIZE_C; s++) dw_val+=accum[s][dv];
        dw_out[base+dv] = dw_val;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── Фаза dk: C[dv,dk]*v[dv] ─────────────────────────────────────────
        for (uint dk=0; dk<HEAD_SIZE_C; dk++) accum[dv][dk] = C_row[dk]*v_dv;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float dk_val=0; for (uint s=0; s<HEAD_SIZE_C; s++) dk_val+=accum[s][dv];
        dk_out[base+dv] = dk_val;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── Фаза da: dsa[dv]*h_prev[dv,dk]  (h_row уже = h_prev) ────────────
        // dsa_dv — локальный регистр, не dsa_sh (barrier убран)
        for (uint dk=0; dk<HEAD_SIZE_C; dk++) accum[dv][dk] = dsa_dv*h_row[dk];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float da_val=0; for (uint s=0; s<HEAD_SIZE_C; s++) da_val+=accum[s][dv];
        da_out[base+dv] = da_val;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── Фаза db: sa_fwd[dv]*C[dv,dk] ────────────────────────────────────
        for (uint dk=0; dk<HEAD_SIZE_C; dk++) accum[dv][dk] = sa_dv*C_row[dk];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float db_val=0; for (uint s=0; s<HEAD_SIZE_C; s++) db_val+=accum[s][dv];
        db_out[base+dv] = db_val;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── Обновление C назад ────────────────────────────────────────────────
        for (uint dk=0; dk<HEAD_SIZE_C; dk++)
            C_row[dk] = C_row[dk]*w_sh[dk] + dsa_dv*a_sh[dk];
    }

    for (uint dk=0; dk<HEAD_SIZE_C; dk++) dh_in_out[h_base+dk] = C_row[dk];
"""
    kern = mx.fast.metal_kernel(
        name=f"wkv7_bwd_v3_H{H}",
        input_names=["r","w","k","v","a","b","h_out_fwd","sa_fwd_in","d_out","d_h_out"],
        output_names=["dr_out","dw_out","dk_out","dv_out","da_out","db_out","dh_in_out"],
        header=hdr, source=body,
        atomic_outputs=False,
    )
    _bwd_cache[H] = kern
    return kern
