import mlx.core as mx

HEAD_SIZE = 64
CHUNK     = 32

_kernel_cache = {}

def _make_kernel(H: int, T: int):
    key = (H, T)
    if key in _kernel_cache:
        return _kernel_cache[key]

    header = f"""
constant uint HEAD_SIZE_C = {HEAD_SIZE};
constant uint CHUNK_C     = {T};
constant uint H_C         = {H};
"""
    source = """
    uint dv   = thread_position_in_grid.y;
    uint bhi  = thread_position_in_grid.x;
    uint bi   = bhi / H_C;
    uint hi   = bhi % H_C;

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
        out[((bi * CHUNK_C + t) * H_C + hi) * HEAD_SIZE_C + dv] = y;
    }

    for (uint dk = 0; dk < HEAD_SIZE_C; dk++) {
        h_out[h_base + dk] = h_row[dk];
    }
"""
    kernel = mx.fast.metal_kernel(
        name         = f"wkv7_H{H}_T{T}",
        input_names  = ["r","w","k","v","a","b","h_in"],
        output_names = ["out","h_out"],
        header       = header,
        source       = source,
    )
    _kernel_cache[key] = kernel
    return kernel


def wkv7_metal(r, w, k, v, a, b, h):
    B, T, H, D = r.shape
    assert D == HEAD_SIZE
    assert T == CHUNK
    r = r.astype(mx.float32)
    w = w.astype(mx.float32)
    k = k.astype(mx.float32)
    v = v.astype(mx.float32)
    a = a.astype(mx.float32)
    b = b.astype(mx.float32)
    h = h.astype(mx.float32)
    kernel = _make_kernel(H, T)
    result = kernel(
        inputs        = [r, w, k, v, a, b, h],
        grid          = (B * H, D, 1),
        threadgroup   = (1, 1, 1),
        output_shapes = [(B, T, H, D), (B, H, D, D)],
        output_dtypes = [mx.float32, mx.float32],
    )
    return result[0], result[1]
