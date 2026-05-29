# HANDOFF — RWKV-7 Metal backward на Apple Silicon / MLX
_Последнее обновление: 29 мая 2026 — полный цикл оптимизации завершён_

---

## Участники и железо

**Алексей**, iOS/macOS разработчик, 18 лет.
**MacBook Air M4**, 16 GB unified RAM, Python `/opt/homebrew/bin/python3`, MLX 0.31.2.

```bash
# GPU wired limit (сбрасывается при перезагрузке):
sudo sysctl iogpu.wired_limit_mb=14336   # 14 GB вместо 12 GB по умолчанию
# Для ctx=4096: обязательно! Для ctx≤2048: желательно.
```

---

## Структура проектов

```
~/Develop/
├── rwkv-mlx/                    ← PRODUCTION: обучение модели
│   ├── train.py                 ← точка входа
│   ├── config.py                ← конфиги моделей
│   ├── model/
│   │   ├── rwkv7.py             ← архитектура RWKV-7
│   │   ├── wkv7.py              ← WKV forward/backward/inference
│   │   └── wkv7_checkpoint.py  ← checkpoint kernel (ТЕКУЩИЙ backward)
│   ├── data/                    ← датасеты .bin
│   └── checkpoints/             ← сохранения
│
└── metal-wkv7/                  ← R&D: разработка и тесты
    ├── wkv7_train_metal.py      ← v2 chunked (baseline, архив)
    ├── wkv7_checkpoint.py       ← checkpoint kernel (ТЕКУЩИЙ)
    ├── wkv7_simd.py             ← эксперимент simd_sum (медленнее)
    ├── wkv7_bwd_v3.py           ← эксперимент bank padding (медленнее)
    ├── wkv7_custom.py           ← Python reference (эталон для тестов)
    ├── wkv7_metal.py            ← inference only
    ├── test_train_metal.py      ← основной тест v2
    └── HANDOFF.md               ← этот файл
```

---

## Текущая конфигурация обучения

```python
# config.py
"debug": ctx_len=1024, batch_size=2   # ← ТЕКУЩИЙ

# train.py
MODEL_DTYPE = mx.bfloat16   # bf16 веса: +5-10% скорость, -12% RAM
```

**Запуск:**
```bash
cd ~/Develop/rwkv-mlx
sudo sysctl iogpu.wired_limit_mb=14336
/opt/homebrew/bin/python3 train.py
```

---

## Исправленные баги

### 1. Утечка памяти → 20 GB своп (критический)
**Симптом:** после ~83000 шагов скрипт занимал 20 GB → агрессивный своп.
**Причина:** `optimizer.update(model, grads)` создаёт lazy-тензоры весов. Без eval
граф накапливается: `w_N → w_{N-1} → ... → w_0`. 83000 шагов → 20 GB.
**Фикс** в `train.py train_step()`:
```python
optimizer.update(model, grads)
mx.eval(model.state, optimizer.state)  # ← КРИТИЧНО, разрывает граф
```

### 2. bf16 backward crash (котангент dtype mismatch)
**Симптом:** `ValueError: Type of cotangents does not match primal output type`
**Причина:** Metal VJP возвращал float32 градиенты для bf16 прималов.
**Фикс** в `wkv7_checkpoint.py`:
```python
grads = [res[0..6]]
return [g.astype(p.dtype) for g, p in zip(grads, primals)]  # cast к dtype примала
```
И в `loss_fn`:
```python
return model.loss(x, y).astype(mx.float32)  # котангент всегда fp32
```

---

## Архитектура WKV kernel (итоговая версия)

### Checkpoint kernel — `wkv7_checkpoint.py`

**Идея:** вместо 16 Python-итераций + 16 mx.eval — ДВА Metal-вызова на весь T.

```
Forward kernel  (один вызов, T токенов):
  for c in 0..N_CHUNKS:
    for t in 0..CHUNK:
      compute forward
    h_checkpoints[c] = h_row   ← сохраняем h после каждых 32 токенов

Backward kernel (один вызов, T токенов):
  C_row = d_h_out
  for c in N_CHUNKS-1..0:
    h_row = h_checkpoints[c]   ← загружаем checkpoint (избегаем взрыв /w)
    for t in CHUNK-1..0:
      ... вычисляем dr,dw,dk,dv,da,db ... (12 барьеров, как v2)
```

**Почему checkpoint необходим для backward:**
Реконструкция `h_prev = (h_cur - v*k - sa*b) / w` усиливает ошибку в (1/w)^steps раз.
При CHUNK=512: (1/0.9)^512 ≈ 10^23 → взрыв. При CHUNK=32: (1/0.9)^32 ≈ 30× → допустимо.

**Математика VJP (на каждый timestep, обратный порядок):**
```
C[dv,dk]  += dy[dv] * r[dk]           # накопление котангента
dv[dv]     = Σ_dk C[dv,dk] * k[dk]   # локально
dsa[dv]    = Σ_dk C[dv,dk] * b[dk]   # локально
dr[dk]     = Σ_dv dy[dv]   * h_cur[dv,dk]   # column sum → shared mem
dw[dk]     = Σ_dv C[dv,dk] * h_prev[dv,dk]  # column sum
dk[dk]     = Σ_dv C[dv,dk] * v[dv]          # column sum
da[dk]     = Σ_dv dsa[dv]  * h_prev[dv,dk]  # column sum
db[dk]     = Σ_dv sa[dv]   * C[dv,dk]       # column sum
h_prev     = (h_cur - v*k - sa*b) / w       # реконструкция
C_prev     = C * w + dsa * a                # обновление C
```

**Технические детали:**
- `grid=(B*H*D, 1, 1)`, `threadgroup=(D=64, 1, 1)`
- `accum[D][D]` = 16 KB threadgroup shared memory для column-sum
- 12 `threadgroup_barrier` на timestep (как в v2, нейтральный рефактор)
- VJP: `mx.eval(h_ckpts, sa_fwd, d_out, d_h_out)` — единственный GPU sync

---

## Полная история экспериментов

### Хронология backward kernel

| Версия | Ключевое изменение | Медиана tok/s | vs v2 | Итог |
|--------|-------------------|---------------|-------|------|
| v2 (baseline) | accum[D][D], 12 барьеров/ts, 16 Python VJP | 18 348 | — | ✅ Baseline |
| simd_sum | simd_sum(), 2 барьера/ts | 17 054 | 0.93× | ❌ Медленнее |
| bank padding | accum[D][D+1] + убраны self-shared | 17 083 | 0.93× | ❌ Медленнее |
| **checkpoint** | 2 Metal-вызова на T, h-checkpoints | **31 000+** | **1.73×** | ✅ PRODUCTION |

> Замеры: B=2, T=32, H=4, D=64, all-6-grads, медиана 40 итераций.

**Почему simd_sum медленнее:** 5×64 = 320 последовательных `simd_sum` на timestep = 1280 циклов.
Экономия барьеров не компенсирует.

**Почему bank padding медленнее:** Apple GPU использует иную организацию threadgroup памяти
чем NVIDIA. Паддинг `[D+1]` нарушает выравнивание вместо устранения конфликтов.

**Checkpoint точнее v2:** v2 имеет data-dependent ошибки 0.3-0.85 для некоторых seed.
Checkpoint стабильно < 3e-5 vs Python reference. Объяснение неизвестно — предположительно
floating point non-commutativity при 16 независимых VJP-вызовах в v2.

### Бенчмарк ctx_len × batch × dtype (checkpoint kernel)

> Все конфиги 2048 tok/step. Медиана 20 итераций. iogpu=14GB.

| ctx | batch | dtype | tok/s | RAM | норм |
|-----|-------|-------|-------|-----|------|
| 512 | 4 | fp32 | 5295 | 11.8G | 0.78 |
| **1024** | **2** | **bf16** | **4997** | **10.3G** | **0.58** ← production |
| 512 | 4 | bf16 | 5841 | 10.5G | 0.77 |
| 1024 | 2 | fp32 | 4761 | 11.7G | 0.58 |
| 2048 | 1 | fp32 | 3811 | 11.6G | 0.45 |
| 2048 | 1 | bf16 | 3941 | 10.4G | 0.44 |
| 4096 | 1 | fp32 | 3986 | 15.4G | 0.36 ⚠️ |
| 4096 | 1 | bf16 | 4073 | 15.3G | 0.35 ⚠️ |

**Ключевые выводы:**
- ctx=512/1024/2048 — ОДИНАКОВАЯ память (11.6-11.8G) при одинаковых tok/step → линейное масштабирование подтверждено
- ctx=4096 = 4096 tok/step (вдвое больше), поэтому и памяти вдвое больше (~15G)
- bf16 даёт +5-10% скорость и -12% RAM при одинаковом качестве (norm идентична)

### Эволюция реальной скорости обучения

| Момент | Конфиг | tok/s | Что изменилось |
|--------|--------|-------|----------------|
| step 83100 | ctx=512 b=4, v2, fp32 | 3 666 | исходное состояние |
| после фикса утечки | ctx=256 b=8, checkpoint, fp32 | 4 500 | fix mx.eval + checkpoint |
| overnight run | ctx=256 b=11, checkpoint, fp32 | 4 575 | batch увеличен |
| **текущий конфиг** | **ctx=1024 b=2, checkpoint, bf16** | **~5000** | **bf16 + длинный ctx** |

---

## Правила бенчмарка

```python
# Прогреть ОБА ядра вместе (≥10 итераций каждое)
for _ in range(10):
    eval(kernel_A); eval(kernel_B)
time.sleep(1)

# Измерять каждую итерацию отдельно
times = [measure_one_iter(kernel) for _ in range(80)]
result = statistics.median(times)   # МЕДИАНА, не mean!
```

Metal компилирует шейдеры двумя проходами (tier-1 быстрый JIT → tier-2 оптимизация фоном).
Без совместного прогрева JIT второго ядра попадает в измерение первого → искажение до 1.5×.
Mean по N прогонам включает JIT-паузы → занижает результат.

---

## Потенциальные оптимизации (от простого к сложному)

### 🟢 Низкая сложность

**1. ctx=4096 как основной контекст** _(1-2 часа)_
При batch=1 bf16 → 4073 tok/s, 15.3G. Требует постоянного iogpu.wired_limit_mb=14336
через /etc/sysctl.conf. Лучшее качество языкового понимания.
Риск: нестабильность если фоновые процессы заберут память.

**2. mx.compile на loss_fn** _(2-4 часа)_
Checkpoint убрал mx.eval из Python-цикла. Теперь единственный eval — в VJP.
mx.compile совместим с mx.custom_function (проверено). Потенциал: +15-30% на
non-WKV операциях (LayerNorm, проекции, softmax через graph fusion).
Блокер: mx.eval внутри VJP может помешать компиляции — нужно проверить.

**3. Сохранение/загрузка чекпоинта в bf16** _(1 час)_
Текущий код сохраняет в fp32. Если сохранять в bf16 → в 2× меньше размер файлов.
При загрузке конвертировать обратно в bf16.

### 🟡 Средняя сложность

**4. Увеличенный batch для ctx=1024** _(3-5 часов)_
ctx=1024 использует 11.7G при batch=2. До 14G влезет batch=3 (оценка ~13.5G).
Проверить: `ctx=1024 batch=3 bf16` — ожидается ~6000-6500 tok/s.
Требует бенчмарка и проверки стабильности памяти.

**5. Gradient checkpointing для модели** _(5-10 часов)_
MLX не имеет встроенного gradient checkpointing для слоёв. Можно реализовать через
mx.custom_function вручную для самых дорогих операций (head projection, 6 TMix блоков).
Потенциал: -30-40% пиковой памяти при обратном проходе → можно увеличить batch.
Риск: +30-50% времени backward (нужна повторная вычисление активаций).

**6. Xcode GPU Frame Capture профилирование** _(2-4 часа)_
Единственный способ точно понять:
- Реальное распределение времени по Metal kernels
- Занятость threadgroup памяти
- Почему bank padding не помог (увидим реальную банковую схему Apple GPU)
- Какие операции занимают больше всего
Инструмент: Xcode → Debug → Capture GPU Frame.

**7. LoRA файнтюн GooseOne 2.9B** _(1 день)_
Модель уже поддерживается в mlx-lm (PR #580 @MollySophia).
```bash
pip install mlx-lm
mlx_lm.lora --model MollySophia/GooseOne-2.9B ...
```
Требует: скачать модель (~6GB), подготовить датасет, настроить LoRA параметры.

### 🔴 Высокая сложность

**8. mx.compile совместимый WKV (mx.scan)** _(1-2 недели)_
mx.scan в MLX 0.31.2 отсутствует. Альтернативы:
a) Дождаться mx.scan в будущих версиях MLX
b) Реализовать собственный scan через mx.custom_function с кастомным VJP
c) Убрать mx.eval из VJP через "deferred eval" паттерн

Разблокирует mx.compile для всего loss_fn → потенциально +30-50% реального обучения.
Самая высокая отдача, самая сложная реализация.

**9. Оптимизация accum column-sum паттерна** _(1-2 недели)_
Текущий backward: 12 threadgroup_barrier на timestep.
accum[D][D] column-sum: каждый поток читает 64 элемента из одного столбца.
Возможные улучшения:
- simdgroup_matrix (AMX) для column-sum: hardware matrix multiply заменяет ручные петли
- Transpose accum в регистры через simd_shuffle_xor
- Слияние фаз dw+da и dk+db в одном accum-проходе (нужны 2 отдельных D×D буфера = 32KB)
  При 128 reg/thread доступно 32KB → теоретически возможно.
Ожидаемый прирост: 1.3-2.0× backward kernel в изоляции.

**10. Custom Metal kernel для head (D→vocab)** _(2-3 недели)_
head projection D=384→V=32000 — самый дорогой matmul в модели.
MLX использует общий GEMM. Специализированный kernel с tile размером под M4 GPU,
BF16 accumulation и fused cross-entropy мог бы дать 1.5-2× на head операции.
Сложность: нужны глубокие знания Metal Performance Shaders и GEMM тюнинга.

**11. 100M/300M модель** _(несколько дней)_
n_embd=768/1024, n_layer=12/24, ctx=4096.
100M: ~8-10 GB, требует iogpu расширения.
300M: ~14-15 GB при batch=1 bf16 — на пределе возможного на 16GB машине.
Предварительный шаг: mx.compile (пункт 8) для экономии памяти на активациях.

---

## Текущий статус всего проекта

| Компонент | Статус |
|-----------|--------|
| Metal forward kernel | ✅ Production |
| Metal backward (checkpoint) | ✅ Production, 1.73× vs v2 |
| Утечка памяти (20 GB) | ✅ Исправлена |
| bf16 обучение | ✅ Работает, +5-10% скорость |
| Оптимальный конфиг (ctx=1024 b=2 bf16) | ✅ Установлен |
| simd_sum backward | ❌ 0.93× — медленнее v2 |
| bank padding (accum[D+1]) | ❌ 0.93× — медленнее v2 |
| mx.compile на loss_fn | ⏳ Не тестировалось с checkpoint |
| ctx=4096 обучение | ⏳ Работает но нужен wired limit |
| mx.scan / compile-compatible WKV | ⏳ mx.scan отсутствует в 0.31.2 |
| 100M/300M модель | ⏳ После mx.compile |
| LoRA файнтюн GooseOne 2.9B | ⏳ Низкий приоритет |

---

## Прогресс обучения debug модели

- **step 83100:** loss 4.9891, 3666 tok/s (v2, ctx=512, b=4, fp32)
- **step 133200:** loss 4.8482, 4575 tok/s (checkpoint, ctx=256, b=11, fp32)
- **текущий конфиг:** ctx=1024, b=2, bf16, ~5000 tok/s, ~10.3 GB
