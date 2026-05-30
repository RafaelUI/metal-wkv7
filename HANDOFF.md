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

## LoRA / QLoRA — РЕАЛИЗОВАНО И ПРОВЕРЕНО (сессия 2026-05-30)

Файлы: `model/lora.py` (движок), тесты `test_lora.py` (если оставлен).
Цель: LoRA/QLoRA-файнтюн через наш WKV Metal kernel на 16 ГБ.

### Что реализовано (model/lora.py)
- `LoRALinear` — обёртка над `nn.Linear`: `y = W·x (frozen) + (alpha/r)·B(A(x))`,
  A ~ N(0,1/√in), B = 0 (ΔW=0 на старте → forward не меняется).
- `add_lora(model, rank, alpha, dropout, tmix_targets, cmix_targets,
  quantize_base, q_group_size, layers)` — оборачивает цели, замораживает базу.
- QLoRA: `quantize_base=4|8` квантует базовую матрицу адаптера; полную базу
  квантуем через `nn.quantize(..., class_predicate=...)` (см. ниже).
- `layers=` — какие блоки получают адаптеры (None=все; range(12,24)=верхние 12).
- `save_lora / load_lora / merge_lora / lora_state`.

### КЛЮЧЕВЫЕ ОТЛИЧИЯ от прежнего плана (важно!)
1. Структура модели — `model.blocks[i].tmix.{r_proj,k_proj,v_proj,o_proj}`,
   НЕ `model.layers[i].attn` как было в эскизе.
2. Заморозка НЕ через `stop_gradient` + `mx.value_and_grad`: обычный
   `mx.value_and_grad` дифференцирует ВСЁ дерево и игнорирует freeze().
   Использовать **`nn.value_and_grad(model, fn)`** — он уважает
   `trainable_parameters()` (freeze()). Подтверждено: дерево градиентов = только
   адаптеры (6.29M на 1.5B, 0.417%, ~12.6 МБ), не-LoRA ключей 0.
3. Gradient checkpointing — ТОЛЬКО `mlx.nn.utils.checkpoint(blk)`.
   Голый `mx.checkpoint` дифференцирует лишь по array-аргументам и ТЕРЯЕТ
   градиент параметров адаптера (loss замирает) — это баг, не способ.

### ЧТО ПРОВЕРЕНО ЭМПИРИЧЕСКИ (M4 16ГБ, ctx512 b2, 1.5B если не указано)

**Корректность обучения (а не просто ненулевые градиенты):**
- forward с адаптером=0 точно = базовый (diff 0) — на random и на реальных весах.
- Градиент r/k/v_proj течёт ЧЕРЕЗ WKV backward kernel (нормы lora_b ненулевые),
  o_proj — после WKV. Все адаптеры получают сигнал.
- full-FT debug-модели → loss 6.23 → 0.0000 за 75 шагов (пайплайн корректен).
- LoRA на ЗАМОРОЖЕННОЙ СЛУЧАЙНОЙ базе почти не учится (6.24→5.67) — ОЖИДАЕМО:
  модулировать нечего. Это НЕ баг.
- LoRA на РЕАЛЬНОЙ предобученной базе (debug ckpt, base loss 4.15):
  loss 4.15 → 2.20 за 125 шагов (overfit 8 реальных батчей) → LoRA учится.
- save/load адаптеров — точный (diff 0). merge_lora → обычный nn.Linear, diff <5e-2.

**mx.compile — ВЫКЛЮЧИТЬ для LoRA:**
- 1.5B LoRA: 43 tok/s (compile) vs 239 tok/s (eager) → в 5.5× МЕДЛЕННЕЕ, 0 выгоды
  по памяти. Steady-state, не разовая компиляция. Скорее всего `nn.value_and_grad`
  + custom_function WKV на 24 слоя не кешируется в графе.

**Gradient checkpointing (nn.utils.checkpoint) — даёт И память И скорость:**
- 1.5B b1: peak 12.84→4.45 ГБ (−2.9×) И 239→533 tok/s (+2.2×). Прежние 239 были
  задушены давлением памяти/свопом; checkpoint снял давление.
- Память почти не растёт с батчем (b1→b2 +0.34 ГБ): хранятся только границы блоков.

**Throughput vs batch — ПЛОСКИЙ ~530-560 tok/s (1.5B):**
- b1=533, b2=564, b8=546, b16=516. Модель "1 батч = 1 ядро GPU" неверна:
  WKV это 4.4% работы, доминируют матмулы, параллельные по (B·T). GPU насыщен
  размерностью T уже при b=1. **Batch — рычаг качества (эфф. batch через grad
  accum), НЕ скорости.** Оптимум скорости — низкий batch.

**QLoRA (4-бит база) — главный рычаг памяти:**
- Только цели 4-бит: active 3.06→2.48 ГБ (мало — цели не основная масса).
- ВСЯ база 4-бит (FFN+head+emb тоже): active 2.48→**0.89 ГБ**, peak 5.71→4.07 ГБ,
  скорость −9% (дектвантизация). → GooseOne 2.9B ≈ 1.5 ГБ active, влезает в 16 ГБ.
- Способ: `nn.quantize(m, group_size=64, bits=4, class_predicate=pred)` где pred
  пропускает цели (их квантует add_lora(quantize_base=4)).
- 8-bit Adam НЕ нужен: для LoRA состояние оптимизатора ~50 МБ (только адаптер).

**Число обучаемых слоёв — рычаг скорости (1.5B QLoRA):**
- все 24: 168 tok/s, peak 4.07 ГБ | верхние 12: 250 (+49%), 3.40 | верхние 6:
  337 (+100%), 3.18 | верхние 3: 404 (+140%), 3.22.
- MLX отсекает backward ниже нижнего адаптера → меньше слоёв = мельче backward =
  быстрее. Память упирается в пол ~3.2 ГБ (forward + head/CE над vocab 65536,
  от backward не зависит). Опустить пол может только chunked CE.

**Память стабильна, утечки НЕТ:**
- Реальное обучение 125+ шагов: active ровно 0.08 ГБ, peak 3.63 ГБ, tok/s ~5780
  стабильно. Баг на 20 ГБ не воспроизводится.
- `mx.set_cache_limit(int(1.5e9))` — кэш Metal-буферов по умолчанию ~2.25 ГБ
  резидента (лечит "своп при свободной RAM"); ограничение до 1 ГБ бесплатно
  (active/peak не меняются).

**Шейдеры НЕ перекомпилируются (проверено инструментально):**
- Наш kernel кешируется по `(H, T)` (имя `wkv7_ckpt_fwd_H{H}_T{T}`), B в grid.
- Фикс. конфиг 30 шагов: ровно 2 компиляции (fwd+bwd), 0 спайков латентности
  после прогрева, steady-state ровный.
- Смена batch → 0 новых компиляций. Смена ctx → +2. **Единственный триггер
  перекомпиляции — непостоянный ctx** (BinDataset держит его фиксированным).
- Xcode Instruments: thermal Nominal (без троттлинга); наша работа = канал
  Compute/Python; Fragment/Vertex в трейсе = графика ОС (WindowServer), не мы.

### РЕКОМЕНДУЕМЫЙ РЕЦЕПТ lora_train.py (всё провалидировано)
- forward через `nn.utils.checkpoint(blk)` по блокам;
- `nn.value_and_grad(model, loss_fn)` (НЕ mx.value_and_grad);
- БЕЗ mx.compile;
- `mx.set_cache_limit(int(1.5e9))` в начале;
- QLoRA: `nn.quantize` всей базы 4-бит + `add_lora(quantize_base=4)`;
- эффективный batch — через grad accumulation (низкий микро-batch);
- `layers=` для компромисса скорость/ёмкость;
- сохранять только адаптеры (`save_lora`).

### Открыто (вне движка)
- chunked cross-entropy над vocab — опускает пол памяти ~3.2 ГБ (нужно для 2.9B
  и больших ctx). Опционально.
- **Конвертация реальных весов RWKV-7 1.5B в наш формат** — ГЛАВНЫЙ ГЕЙТ.
  Риск: наша rwkv7.py может не совпадать тензор-в-тензор с официальной 1.5B
  (ln_x: у нас nn.LayerNorm — у официальной GroupNorm по головам; token-shift;
  k_k/k_a/r_k; v_first). Сверять построчно ДО конвертации.
- Инструкционные данные для перефразирования (JSONL пар).

---

## Конвертация официальной RWKV-7 World 1.5B — ПРОЙДЕНО (2026-05-30)

Файлы (в rwkv-mlx): model/rwkv7_x070.py (x070-точная архитектура),
convert_rwkv_pth.py (torch-free загрузчик .pth + маппинг),
model/world_tokenizer.py (TRIE World), tokenizer/rwkv_vocab_v20230424.txt,
веса weights/rwkv7_1.5B_x070.safetensors (2.8 ГБ bf16).

ИТОГ: официальная World 1.5B грузится в наш формат и РАБОТАЕТ.
- 795 ключей модели == 795 конвертера, 0 расхождений форм.
- loss на реальном русском тексте = 3.26 (ppl 26), random=11.09 → КОРРЕКТНО.

Почему отдельный файл: rwkv7.py (from-scratch, 32k RU токенайзер) оставлен как есть
(debug-чекпоинт под него валиден). rwkv7_x070.py — каноничная версия под официальные веса.

6 архитектурных фиксов в rwkv7_x070 относительно rwkv7.py:
 1. decay: +w0 (bias w_lora_B) внутри sigmoid
 2. iclr a: без tanh
 3. gate g: sigmoid ВНУТРИ, линейно наружу (без gelu/внешнего sigmoid)
 4. ln_x: GroupNorm по головам (num_groups=H, eps=64e-5), не LayerNorm
 5. порядок: WKV -> ln_x -> +bonus (а не +bonus -> ln_x)
 6. token-shift: нулевой паддинг t=0 в каждом блоке, без межблочного переноса; cmix шифтует свой вход
 + low-rank ранги по формуле из D (1.5B: w=a=96, v=64, g=256), не хардкод 64.

Маппинг (официал x070 -> наш): проекции/emb/head/ffn/ln напрямую (официал хранит
(out,in) как nn.Linear); low-rank w1/w2/a1/a2/v1/v2/g1/g2 ТРАНСПОНИРОВАТЬ;
k_k/k_a (1,1,D)->(H,S); r_k уже (H,S); w0/a0/v0 (1,1,D)->bias (D); v* только слои>0.

torch на Mac НЕ нужен: torch-free ридер .pth (zip+pickle, persistent_load -> MLX),
т.к. mlx-python 3.14 без torch-колёс. bf16 читается как uint16 -> view(bfloat16).

ВАЖНО — токенайзеры НЕсовместимы: train.bin = кастомный 32k RU (для from-scratch rwkv7.py),
официальная модель = World 65536. Для LoRA официальной нужна World-токенизация
(cleaned.txt НЕ удалять — исходник под World-данные).

Дальше: (1) lora_train.py по провалидированному рецепту, (2) инструкционные данные
(JSONL пар перефразирования, World-токенизация), (3) 2.9B — после чистки диска (~6+6 ГБ).

## Капстоун: LoRA на реальной World 1.5B — ПРОЙДЕНО (2026-05-30)

Весь продакшен-стек проверен на НАСТОЯЩЕЙ модели (capstone_lora_real.py в rwkv-mlx):
официальная World 1.5B -> конвертер -> QLoRA 4-бит база -> LoRA (верхние 12 слоёв) ->
nn.utils.checkpoint -> World-токенайзер -> реальный русский текст.
- baseline bf16 = 3.08, 4-бит = 3.02 (квантизация НЕ испортила).
- overfit 6 окон: 3.02 -> 0.006 ПЛАВНО, без взрыва. peak 4.40 ГБ, 250 tok/s.

КРИТИЧНЫЙ УРОК — гиперпараметры для ПРЕДОБУЧЕННОЙ базы:
- lr=2e-3, alpha=32 (scale 2.0) -> РАСХОДИМОСТЬ (3.02 -> 13.9 -> 6.2). Сильную модель
  агрессивные LoRA-апдейты разносят мгновенно.
- lr=1e-4, alpha=16 (scale 1.0) -> гладкая сходимость к 0.
- На debug-базе грубые lr проходили лишь потому, что она слабая/недообученная.
- Вывод: для файнтюна официальной модели стартовать с lr~1e-4, alpha=16, clip 1.0.

QLoRA-нюанс x070: nn.quantize НЕЛЬЗЯ применять ко всем nn.Linear — внутренние
low-rank матрицы (w/a/g/v: ранги 96/256/64/32) не кратны group_size 64 -> падение.
Квантовать ТОЛЬКО большие замороженные: cmix.key/value, head, emb (+ r/k/v/o через
add_lora). Low-rank оставлять bf16 (память копеечная, точность динамики не портим).

ENGINE ГОТОВ. Дальше — фаза данных/файнтюна:
 (1) lora_train.py: рецепт = nn.utils.checkpoint forward, nn.value_and_grad, без compile,
     set_cache_limit(1.5e9), QLoRA (big-only quant), lr 1e-4/alpha16/clip1.0, grad accum,
     save_lora; layers= для скорость/ёмкость.
 (2) данные перефразирования: JSONL пар из cleaned.txt, World-токенизация.
 (3) 2.9B — после чистки диска.

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
| LoRA / QLoRA движок | ✅ Реализован и проверен на реальном обучении |
| 138.9M предобучение | ⏳ ~33 дня (нецелесообразно) |
| Конвертация официальной 1.5B | ✅ Загружена, loss 3.26 (ppl 26) |
| Продакшен-стек на реальной 1.5B | ✅ Капстоун: 3.02->0.006 |
| GooseOne LoRA файнтюн | ⏳ Фаза данных |
| chunked cross-entropy | ⏳ Опционально (пол памяти) |

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
