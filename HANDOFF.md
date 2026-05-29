# HANDOFF — RWKV-7 Metal Training на Apple Silicon / MLX
_Последнее обновление: 29 мая 2026 — полный цикл оптимизации + inference_

---

## Участники и железо

**Алексей**, iOS/macOS разработчик, 18 лет.
**MacBook Air M4**, 16 GB unified RAM, Python `/opt/homebrew/bin/python3`, MLX 0.31.2.

```bash
# GPU wired limit (сбрасывается при перезагрузке):
sudo sysctl iogpu.wired_limit_mb=14336
# Постоянно: добавить в /etc/sysctl.conf
```

---

## Структура проекта (production)

```
~/Develop/
├── rwkv-mlx/                    ← PRODUCTION: обучение с нуля
│   ├── train.py                 ← точка входа
│   ├── config.py                ← конфиги моделей
│   ├── model/
│   │   ├── rwkv7.py             ← архитектура RWKV-7
│   │   ├── wkv7.py              ← WKV forward/backward/inference
│   │   └── wkv7_checkpoint.py  ← checkpoint Metal kernel (production)
│   ├── data/                    ← датасеты .bin (gitignored)
│   └── checkpoints/             ← сохранения (gitignored)
│
└── metal-wkv7/                  ← R&D: kernel разработка
    ├── wkv7_checkpoint.py       ← основной kernel
    ├── wkv7_custom.py           ← Python reference
    ├── wkv7_metal.py            ← inference kernel
    ├── test_full.py             ← тесты
    ├── test_isolate.py
    └── experiments/             ← провальные эксперименты (simd, v3)
```

---

## Текущий конфиг обучения

```python
# config.py
"debug": RWKVConfig(n_layer=6, n_embd=384, vocab_size=32000, ctx_len=1024, batch_size=4)

# train.py
CFG_NAME    = "debug"
MODEL_DTYPE = mx.bfloat16
GRAD_ACCUM  = 1          # для debug: 1 (нет аккумуляции)
                          # для 138.9M: 4 (b=4 ctx=512)
```

**Запуск:**
```bash
cd ~/Develop/rwkv-mlx
sudo sysctl iogpu.wired_limit_mb=14336
/opt/homebrew/bin/python3 train.py
```

---

## История всех оптимизаций (хронологически)

### 1. Исправление утечки памяти → 20 GB своп
**Симптом:** step 83100, скрипт занимал 20 GB после длительного обучения.
**Причина:** `optimizer.update()` создаёт lazy-тензоры без материализации.
За 83000 шагов: `w_N → w_{N-1} → ... → w_0` — цепочка в 20 GB.
**Фикс:**
```python
optimizer.update(model, grads)
mx.eval(model.state, optimizer.state)  # ← разрывает lazy граф
```

### 2. Metal WKV7 v2 (chunked backward)

Python einsum → CUDA-style chunked Metal kernel.
- Реализован `torch.autograd.Function`-аналог через `mx.custom_function`
- VJP считает все 6 градиентов (dr, dw, dk, dv, da, db)
- 16 Python-итераций по CHUNK=32 токенов
- Результат: **3 666 tok/s** (+4.1× vs Python ~900)

### 3. Metal WKV7 Checkpoint Kernel (основное ускорение)

**Идея:** заменить 16 Python VJP-вызовов на 2 Metal-вызова для всего T.

**Forward** (один kernel, T токенов):
```metal
for (uint c=0; c<N_CHUNKS; c++) {
    for (uint t=0; t<CHUNK; t++) { /* WKV step */ }
    h_checkpoints[c] = h_row;  // ← сохраняем h каждые 32 токена
}
```

**Backward** (один kernel, обратный порядок):
```metal
for (int c=N_CHUNKS-1; c>=0; c--) {
    h_row = h_checkpoints[c];  // ← загружаем точный checkpoint
    for (int t=CHUNK-1; t>=0; t--) { /* VJP step */ }
}
```

**Зачем checkpoint:** реконструкция `h_prev = (h_cur - ...) / w` усиливает ошибку в
`(1/w)^steps`. При CHUNK=512: `(1/0.9)^512 ≈ 10^23` → взрыв. При CHUNK=32: ×30 → OK.

**Технические детали:**
- `grid=(B*H*D, 1, 1)`, `threadgroup=(D=64, 1, 1)`
- `accum[D][D]` = 16 KB shared memory для column-sum транспонирования
- 12 `threadgroup_barrier` на timestep (неизбежно для корректности)
- Единственный `mx.eval` в VJP → 1 GPU sync vs 32 в v2

**Корректность vs Python:** max diff < 2.5e-5 ✓
**Численная точность:** checkpoint точнее v2 для некоторых seed (v2 имеет
data-dependent ошибки 0.3-0.85 из-за накопления в float32 при 16 независимых VJP)

**Результат:** **5 000 tok/s** (+1.73× vs v2)

### 4. bf16 обучение

**Проблема:** VJP возвращал fp32 градиенты для bf16 прималов → crash.
**Фикс:**
```python
# wkv7_checkpoint.py VJP:
grads = [res[0]..res[6]]
return [g.astype(p.dtype) for g, p in zip(grads, primals)]  # ← dtype cast

# train.py:
return model.loss(x, y).astype(mx.float32)  # ← котангент всегда fp32
```

**Результат:** **~6 050 tok/s** (+12%), -12% RAM, checkpoint files вдвое меньше
(MLX save_weights сохраняет в dtype модели → bf16 auto)

### 5. mx.compile

**Проблема:** `mx.eval` внутри VJP запрещён в контексте compile.
**Фикс:** убрать `mx.eval(h_ckpts, sa_fwd, d_out, d_h_out)` из VJP — Metal kernel
принимает lazy tensors и материализует их сам при dispatch.

**Реализация:**
```python
def make_train_step(model, optimizer, grad_accum=1):
    if grad_accum == 1:
        state = [model.state, optimizer.state]
        def _step(x, y):
            loss, grads = mx.value_and_grad(loss_fn)(model, x, y)
            grads, norm = optim.clip_grad_norm(grads, max_norm=1.0)
            optimizer.update(model, grads)
            return loss, norm
        return mx.compile(_step, inputs=state, outputs=state)
    # ... (accum path ниже)
```

**Почему compile экономит память:** kernel fusion убирает промежуточные тензоры.
Без compile: каждая операция (layernorm, sigmoid, lerp...) = отдельный тензор в RAM.
С compile: все element-wise операции сливаются в один Metal kernel → промежуточные
значения живут только в регистрах GPU.

**Результат:**
- Скорость: **+12%** (6 050 → 6 720 tok/s при batch=4)
- Память: **-2.8×** на батч (2.625 GB → 0.95 GB/batch item)
- Это разблокировало batch=12 (был batch=4): **6 978 tok/s**

### 6. Gradient Accumulation

**Для 138.9M модели:** без аккумуляции batch=4 ctx=512 → OOM.
С аккумуляцией: N микро-шагов, один optimizer update.

**Критический баг (при реализации):**
```python
# НЕПРАВИЛЬНО — 28 GB взрыв:
total_grads = tree_map(lambda a,b: a+b, total_grads, grads_i)
# lazy граф: g1 + g2 + g3 + g4 → держится в памяти всё дерево

# ПРАВИЛЬНО:
mx.eval(grads_i)        # ← материализуем до сложения
total_grads = tree_map(lambda a,b: a+b, total_grads, grads_i)
mx.eval(total_grads)    # ← материализуем накопленное
```

**compiled_micro:** каждый микро-шаг компилируется отдельно:
```python
micro_state = [model.state]  # веса не меняются между микро-шагами
def _micro_fn(x, y):
    return mx.value_and_grad(loss_fn)(model, x, y)
compiled_micro = mx.compile(_micro_fn, inputs=micro_state)
```

---

## Полная таблица скоростей

| Версия | tok/s | Что добавлено |
|--------|-------|---------------|
| Python einsum | ~900 | исходное |
| Metal v2 chunked | 3 666 | Metal VJP |
| Checkpoint kernel | ~5 000 | 2 вызова на T |
| + bf16 | ~6 050 | dtype cast fix |
| + mx.compile | 6 720 | kernel fusion |
| + **batch=12** | **6 978** | compile → -2.8× RAM |
| **Итого: 7.8× vs Python** | | |

---

## Бенчмарки по конфигам (checkpoint + bf16 + compile)

### debug 36.4M

| ctx | batch | dtype | tok/s | RAM |
|-----|-------|-------|-------|-----|
| 512 | 4 | fp32 | 5295 | 11.8G |
| **1024** | **4** | **bf16** | **~6000** | **10.5G** ← production |
| 2048 | 1 | bf16 | 3941 | 10.4G |
| 4096 | 1 | bf16 | 4073 | 15.3G ⚠️ |

ctx=512, 1024, 2048 имеют одинаковую память при одинаковом tok/step —
линейное масштабирование checkpoint kernel подтверждено.

### 138.9M (с compiled_micro)

| Конфиг | tok/s | RAM |
|--------|-------|-----|
| b=4 ctx=512 accum=4 | **1739** | 12.1G ✅ |
| b=4 ctx=512 accum=2 | 1711 | 11.8G ✅ |
| b=2 ctx=1024 accum=2 | 1634 | 11.7G ✅ |

**Потолок 138.9M на M4 Air:** ~2 000 tok/s (8× больше FLOPs чем debug).
5B токенов при 1739 tok/s = ~33 дня. Нецелесообразно для предобучения.

---

## Провальные эксперименты

| Эксперимент | Результат | Причина |
|-------------|-----------|---------|
| simd_sum backward | 0.93× медленнее | 320 последовательных simd_sum = 1280 циклов |
| accum[D][D+1] bank padding | 0.93× медленнее | Apple GPU ≠ NVIDIA banking |
| ANE для matmul | не реализовано | WKV нельзя выразить на ANE без hit лимита |
| bf16 без dtype cast | crash | cotangent mismatch |
| tree_map без eval | 28 GB взрыв | lazy граф |

---

## Профилирование (Xcode Instruments Metal System Trace)

| Метрика | Значение |
|---------|----------|
| GPU утилизация Python | 77.8% |
| CPU→GPU latency | 80.72ms |
| Dispatches (Python) | 6 744 за 58с |
| CustomKernel (WKV) | **4.4%** работы |
| Matmul | **22%** работы |
| Compile fused ops | 23.7% работы |

**Вывод:** WKV = 4.4% → дальнейшая оптимизация WKV бессмысленна.
Bottleneck: matmul (compute-bound при batch≥4), optimizer bandwidth.

### Арифметическая интенсивность операций

| Операция | FLOPs/byte | Тип |
|----------|-----------|-----|
| Проекция (B×T,D)×(D,D) | 204 | compute-bound |
| FFN (D→4D→D) | ~200 | compute-bound |
| Head (D→vocab) | 478 | compute-bound |
| **WKV (sequential)** | **~1** | **bandwidth-bound** ← поэтому Metal помогло |

---

## Apple Neural Engine (ANE) — анализ

| Параметр | Значение | Источник |
|----------|----------|---------|
| Реальная производительность | 19 TOPS INT8 | эксперименты |
| FP16 throughput | ~15.8 TFLOPS | Orion paper |
| Bandwidth | 60 GB/s | эксперименты |
| SRAM | ~32 MB | Orion/maderix |
| Лимит компиляций | 148/фаза | эксперименты |
| Параллельность с GPU | ДА | эксперименты |

**Почему ANE не помогает для RWKV training:**
- При ctx=128: GPU compile даёт 0.7ms, ANE для SmolLM 7ms (GPU быстрее!)
- RWKV линейный по T → GPU с compile уже оптимален
- WKV recurrence нельзя выразить без разворачивания цикла

**Где ANE полезен:** inference на устройстве (CoreML RWKV для iPhone уже существует,
0.4B даёт 100 tok/s на iPhone). Для нашего проекта: inference финальной debug модели.

---

## 138.9M Inference

```
RWKV-7 World3 1.5B (L24-D2048-H32) через mlx-lm:
  Память: 9.7 → 10.8 GB
  Скорость: ~25-30 tok/s
  Токенизатор: RWKV World (65536 токенов), НЕ GPT-2 (50254)
  Формат промпта: "User: ...\n\nAssistant:"
```

**Проблемы при загрузке:**
1. RWKV/RWKV7-Goose-World3-1.5B-HF требует `fla` пакет (нет на PyPI)
2. tokenizer.json = GPT-2 vocab (50254), неправильный для World модели
3. Решение: патчить config.json (убрать auto_map), использовать RWKV World tokenizer
   из `/opt/homebrew/lib/python3.14/site-packages/rwkv/rwkv_vocab_v20230424.txt`

**Для полноценного inference:**
- Base World модель не следует русским инструкциям
- Нужна instruction-tuned версия (GooseOne G1c)
- Или LoRA файнтюн на инструкционных парах

---

## Следующий шаг: LoRA через наш Metal kernel

### Архитектура LoRA для RWKV-7

```
W_frozen + ΔW = W_frozen + α/r × B × A
  B: D → rank   (обучаем)
  A: rank → D   (обучаем)
  W: заморожен  (gradient.stop_gradient)
```

**Где ставить адаптеры:**
- `receptance.weight` (r_proj) → градиент течёт через WKV backward ← наш kernel
- `key.weight` (k_proj)        → через WKV backward
- `value.weight` (v_proj)      → через WKV backward
- `output.weight` (o_proj)     → НЕ через WKV (после WKV)
- FFN up/gate/down             → НЕ через WKV

**Память для GooseOne 2.9B LoRA (rank=16):**
- Веса base (bf16): ~5.8 GB
- LoRA adapter: ~8 MB
- Optimizer state: ~32 MB (только адаптер!)
- Итого: ~7-8 GB → влезает без аккумуляции

### Реализация (план)

```python
class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, rank: int = 16, alpha: float = 32.0):
        self.linear = linear   # заморожен
        self.A = nn.Linear(linear.weight.shape[1], rank, bias=False)
        self.B = nn.Linear(rank, linear.weight.shape[0], bias=False)
        self.scale = alpha / rank
        # Инициализация: A ~ N(0, σ), B = 0 → ΔW = 0 на старте
        self.A.weight = mx.random.normal(self.A.weight.shape) * 0.02
        self.B.weight = mx.zeros(self.B.weight.shape)

    def __call__(self, x):
        return mx.stop_gradient(self.linear(x)) + self.scale * self.B(self.A(x))

def add_lora(model, rank=16, target_modules=('r_proj', 'k_proj', 'v_proj', 'o_proj')):
    for layer in model.layers:
        for name in target_modules:
            if hasattr(layer.attn, name):
                orig = getattr(layer.attn, name)
                setattr(layer.attn, name, LoRALinear(orig, rank))
    return model
```

**Данные для файнтюна под перефразирование русской литературы:**
1. Взять 1000 абзацев из своего датасета литературы
2. Сгенерировать пары через Claude API (~$2-3)
3. Формат JSONL: `{"instruction": "Перефразируй: ...", "output": "..."}`

---

## Статус компонентов

| Компонент | Статус |
|-----------|--------|
| Metal forward kernel | ✅ Production |
| Metal backward (checkpoint) | ✅ Production, 1.73× vs v2 |
| Утечка памяти | ✅ Исправлена |
| bf16 обучение | ✅ Работает |
| mx.compile | ✅ Работает |
| Gradient accumulation | ✅ Работает (compiled_micro) |
| bf16 checkpoints | ✅ Автоматически (save_weights) |
| simd_sum backward | ❌ 0.93× — медленнее |
| bank padding | ❌ 0.93× — медленнее |
| RWKV-7 1.5B inference | ✅ 25-30 tok/s, 10.8 GB |
| LoRA реализация | ⏳ Следующая сессия |
| 138.9M предобучение | ⏳ ~33 дня (нецелесообразно) |
| GooseOne LoRA файнтюн | ⏳ Следующая сессия |

---

## Прогресс обучения debug модели

| Step | Loss | tok/s | Конфиг |
|------|------|-------|--------|
| 83100 | 4.9891 | 3 666 | v2, ctx=512, b=4, fp32 |
| 133200 | 4.8482 | 4 575 | checkpoint, ctx=256, b=11, fp32 |
| 157050 | 4.5551 | 6 978 | checkpoint+compile+bf16, ctx=1024, b=12 |
| 157500 | 4.7610 | 5 918 | ctx=1024 b=4 (тест нового конфига) |

---

## Правила бенчмарка

```python
# 1. Прогреть ОБА ядра вместе (≥10 итераций каждое)
for _ in range(10): eval(A); eval(B)
time.sleep(1)
# 2. Медиана, не mean (mean занижает из-за JIT-пауз)
times = [measure() for _ in range(80)]
result = statistics.median(times)
```
