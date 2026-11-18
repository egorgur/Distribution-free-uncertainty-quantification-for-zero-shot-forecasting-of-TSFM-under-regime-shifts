# Distribution-free-uncertainty-quantification-for-zero-shot-forecasting-of-TSFM-under-regime-shifts

Исследование адаптации замороженных Time-Series Foundation Models к смене режима без дообучения весов. В переходах могут ухудшаться точечный прогноз и покрытие интервалов; форма деградации зависит от типа смены. Старая история не обязательно бесполезна: выполненные эксперименты показали, что простая обрезка не улучшает точечный прогноз в среднем.

## Актуальный план v3 — 2026-11-18

По [инструкции по дальнейшей работе](Инструкция%20по%20дальнейшей%20работе.docx) исследуются четыре направления:

1. **Multi-Context:** адаптивный выбор или смешивание прогнозов по L64/128/256/512.
2. **Regime-Aligned Context:** коррекция уровня, масштаба или тренда старой истории; выбор через прогноз назад по известным наблюдениям.
3. **Regime-Component Adaptation:** определение изменившейся компоненты и специализированная реакция, включая No Adaptation.
4. **Adaptive Uncertainty:** коррекция ширины/центра интервалов, затем последовательная conformal-коррекция и несколько масштабов памяти.

Полная постановка, H7–H9, метрики и порядок работы — [план адаптации](docs/adaptation_plan.md). Правила и текущий статус — [CLAUDE.md](CLAUDE.md), термины — [GLOSSARY.md](GLOSSARY.md), решения — [журнал](docs/decisions.md).

**Ближайший шаг:** согласовать протокол небольшого синтетического пилота Multi-Context, затем переиспользовать подходящие готовые кеши. Конкретные веса, hindcast, критерии, нормировка и бюджет ещё не утверждены. Четыре новых метода пока не реализованы; команды ниже воспроизводят существующую инфраструктуру v2, а не запускают адаптацию v3.

Веса TSFM остаются замороженными во всех направлениях. Направление 4 использует уже созревшие ошибки целевого ряда и является **внешней онлайн-калибровкой**: прежнее условие «без целевой калибровки» не относится ко всей v3. Формальных distribution-free/conformal-гарантий для зависимых данных со сменами не заявляем.

### Что уже получено

На синтетике выполнены анализ риска/покрытия, сравнение сигналов надёжности, локализация и каскад. H5 остаётся отрицательной по прежнему критерию; новые гипотезы не проверены. Выводы — [этап 3](docs/results/stage3_failures.md), [этап 4](docs/results/stage4_signals.md), [этап B](docs/results/stage5_localization.md), [каскад](docs/results/stage6_cascade.md). [Протокол H1/H3 v1](docs/protocol_h1_h3.md) и старые результаты сохраняются без переопределения.

Реальных прогнозов ещё нет: аудиты шести кандидатов не установили готовый строгий пилот с устойчивыми эталонами. Это ограничение не отменено, но больше не блокирует первый синтетический эксперимент адаптации. Отчёт на 13 ноября — [исторический срез](docs/interim_status_2026-11-13.md); актуальная очередь — план v3.

## Установка

Локально, на CPU (нужен [uv](https://docs.astral.sh/uv/)):

```bash
uv venv .venv --python 3.13
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python -r requirements.txt
```

Локально, с GPU NVIDIA. torch 2.15.0 с CUDA выпущен только под CUDA 13.2 и 13.4, а там нет поддержки старых карт (Pascal, например GTX 10xx). Для них — torch 2.14.1 под CUDA 12.6, остальные пакеты как в `requirements.txt`:

```bash
uv venv .venv --python 3.13
uv pip install --python .venv/bin/python torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
grep -v '^torch==' requirements.txt > /tmp/req.txt && echo 'torch==2.14.1+cu126' > /tmp/torch.txt
uv pip install --python .venv/bin/python -r /tmp/req.txt --constraint /tmp/torch.txt \
    --extra-index-url https://download.pytorch.org/whl/cu126 --index-strategy unsafe-best-match
.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_arch_list())"
```

В Google Colab (torch уже установлен; пока не проверялось), в ячейке ноутбука:

```
!git clone https://github.com/egorgur/Distribution-free-uncertainty-quantification-for-zero-shot-forecasting-of-TSFM-under-regime-shifts.git repo
%cd repo
!pip install chronos-forecasting==2.3.2
!python -m scripts.check_chronos --device cuda
```

Проверка, что Chronos-Bolt работает (при первом запуске скачивает веса, ~190 МБ):

```bash
python -m scripts.check_chronos                                    # chronos-bolt-small, CPU
python -m scripts.check_chronos --model amazon/chronos-bolt-base   # другой размер модели
```

## Реальные данные: газовый стенд UCI

Загрузка двух исходных TXT и архива в `data/raw/uci_gas_dynamic/`, затем полный аудит без запуска моделей:

```bash
python -m scripts.download_gas_data
python -m scripts.audit_gas_data
python -m scripts.check_gas_audit
python -m unittest discover -s tests
```

Нужно около 1.6 ГБ на исходники с архивом и 20 МиБ на аудит. Каждый TXT больше лимита GitHub в 100 МБ, поэтому ZIP/TXT исключены из Git; источник и контрольные суммы — в `data/raw/uci_gas_dynamic/manifest.json`. При повторном запуске загрузчик проверяет имеющиеся файлы, а не перезаписывает их отличающимися данными.

Полные таблицы и графики — `results/real_gas_audit/`, выводы и ограничения — `docs/results/real_gas_audit.md`. Частота, предобработка и протокол реальных H1–H3 ещё не утверждены. Секундные агрегаты аудита — только для диагностики, не выбранная модельная сетка. Условия использования и столбцы — `data/raw/uci_gas_dynamic/README.md`.

## Аудит SKAB и TEP MIT

Загрузчик `python -m scripts.download_tep_skab` сохраняет оригиналы в `data/raw/skab/` и `data/raw/tep_mit/`, закрепляет SKAB commit и сверяет Git blob/ZIP CRC. Числовые исходники исключены из Git; манифесты сохраняются. TEP — симуляция, не реальные измерения; расширенные RData Rieth сюда не входят.

Полный аудит без моделей: `python -m scripts.audit_skab` и `python -m scripts.audit_tep`. Повторная проверка таблиц/оригиналов: те же команды с `--check`. Все тесты: `python -m unittest discover -s tests`.

Выводы — `docs/results/skab_audit.md`, `docs/results/tep_audit.md`; таблицы и сверки — `results/skab_audit/`, `results/tep_audit/`. SKAB требует отдельного правила нерегулярного времени и группировки дубликатов; у TEP L512 не позволяет наблюдать окрестность вмешательства. Готовый прогнозный протокол из этих аудитов не следует.

### Предварительный экран эталонов UCI gas

Правила описательной проверки — `docs/protocol_gas_reference_audit.md`, выводы и ограничения мощности — `docs/results/gas_reference_audit.md`. Выполнение без TSFM: `python -m scripts.audit_gas_reference`; проверка — та же команда с `--check`. Повторный прогон с контролями сохраняется в `results/gas_reference_audit_v1_recheck/`, первоначальный снимок — отдельно в `results/gas_reference_audit_v1/`. Ноль кандидатов не доказывает непригодность данных: стабильный контроль тоже не прошёл. Устойчивый знаменатель `mae_ratio` не утверждён.

Полная синтетическая проверка мощности выполнена отдельно: `docs/results/reference_screen_power.md`, результаты — `results/reference_screen_power_v1/`. Проверка сохранённого расчёта: `python -m scripts.check_reference_screen_power --check`. Новое воспроизведение (около 2 мин CPU) требует пустого каталога: `python -m scripts.check_reference_screen_power --out results/reference_screen_power_repeat`. Исходный экран не принимает даже стационарные AR-контроли; его не следует использовать для автоматического обоснования эталонов. Допуски ради ненулевого отбора не менялись.

## Дополнительный кандидат: PRONTO

Без большого ZIP/видео/HF можно получить журнал и один дневной CSV:
`python -m scripts.audit_pronto_documents`, затем `python -m scripts.audit_pronto_test11`.
Проверка каждого сохранённого снимка — соответствующая команда с `--check`.
Актуальные документы — `data/raw/pronto_documents_v2/`, Test11 —
`data/raw/pronto_test11/`; результаты — `results/pronto_documents_v2/`,
`results/pronto_test11/`. Отчёт — `docs/results/pronto_reference_audit.md`.
Это структурный аудит, не утверждённая модельная сетка или эталон H1–H3.
Полный ZIP не скачивается, его MD5 не проверяется; CRC отдельных членов и
локальные SHA256 сохраняются. Первоначальный документальный снимок v1 — исторический.

Все дневные SCADA: `python -m scripts.download_pronto_scada`, затем
`python -m scripts.audit_pronto_scada`; повторная сверка — последняя команда
с `--check`. Проверены10архивных членов, четыре уникальных дня; CSV локальны в
`data/raw/pronto_scada/`, актуальные таблицы — `results/pronto_scada_v2/`.
Отчёт — `docs/results/pronto_daily_audit.md`: при L512/H64 пара длинных устойчивых
старого/нового эталонов не установлена. Инференс не запускался.

## Ограниченный аудит UCI447 и следующий кандидат

`python -m scripts.audit_uci447` получает оригинальный ZIP (~76.6МБ, локально,
исключён изGit), проверяет профиль и один100Гц каналPS1 без склейки циклов.
Повторная сверка — та же команда с `--check`. Манифест/исходные метаданные —
`data/raw/uci447/`, результаты — `results/uci447_audit_v1/`, выводы —
`docs/results/uci447_audit.md`. Остальные16каналов численно не аудированы,
устойчивые эталоны не установлены. Все190тестов проекта прошли.

Следующий документально проверенный кандидат — MCC5-THU Motor:
`docs/results/mcc5_source_audit.md`. Продолжение уже выполнено:
`docs/results/mcc5_one_run_audit.md`. Через Range получен ровно один полныйCSV,
оригинал119МБ исключён изGit, манифесты/первичные документы сохранены.
1 152 000строк,9столбцов с временем; устойчивые эталоны/единицы момента
пока не установлены. Большой архив целиком не получен, прогнозов нет.

Уточнение версий статьи/архивов и репозитория соавтора:
`docs/results/mcc5_source_followup.md`. Подготовлен адресный запрос авторам
`docs/mcc5_author_request.md`, **не отправлен**. Это документальное
продолжение; исходные результаты/код/212тестов не менялись.

Проверка источника — `python -m scripts.download_mcc5 --check`.
Проверка актуального аудита — `python -m scripts.audit_mcc5 --out results/mcc5_one_run_v2 --check`.
Новый расчёт требует пустого каталога: `python -m scripts.audit_mcc5 --out results/mcc5_repeat`.
Первый `mcc5_one_run_v1` оставлен историческим после уточнения типов;
все12продуктов совпали побайтово. Все212тестов проекта прошли.

Для повторной загрузки сначала `python -m scripts.download_mcc5 --inventory-only`,
затем фиксация точного выбранного пути до сигналов и `python -m scripts.download_mcc5`.
Ограничения — `docs/protocol_mcc5_bounded_audit.md`; это не разрешение
перебирать другие файлы ради эталона или автоматически менять частоту/L/H.

## Синтетика и кеш прогнозов

Сетка факторов задана в `configs/synth_v1.yaml`. Примеры рядов каждого типа сохраняются в `results/synth_v1/examples.png`:

```bash
python -m scripts.plot_synth
```

Прогнозы всех окон и всех вариантов контекста (S4, S7) сохраняются в `results/cache/synth_v1/`. Кеш хранится в git (~1.1 ГБ), поэтому после `git clone` его пересчитывать не нужно. Если прогон прервался, та же команда продолжит его с места остановки:

```bash
python -m scripts.cache_forecasts --model amazon/chronos-bolt-small   # ~1 ч на CPU
python -m scripts.cache_forecasts --n-per-cell 1                      # пробный прогон, results/cache/synth_v1_n1/
```

Chronos-2 на CPU ноутбука считает полную сетку около 10 ч, поэтому её удобнее считать в Colab с GPU (пока не проверялось). Кеш пишется на Google Drive, чтобы пережить отключение сессии; после отключения та же команда продолжит расчёт. Код берётся из GitHub, поэтому сначала нужно отправить туда локальные изменения (`git push`). В ячейке ноутбука (Runtime → Change runtime type → GPU):

```
from google.colab import drive; drive.mount('/content/drive')
!git clone https://github.com/egorgur/Distribution-free-uncertainty-quantification-for-zero-shot-forecasting-of-TSFM-under-regime-shifts.git repo
%cd repo
!pip install chronos-forecasting==2.3.2
!python -m scripts.check_chronos --model amazon/chronos-2 --device cuda
!python -m scripts.cache_forecasts --model amazon/chronos-2 --device cuda --out /content/drive/MyDrive/tsfm_cache/synth_v1
```

Ряды в Colab генерируются заново по конфигу. Прежде чем класть скачанный кеш рядом с локальным, нужно убедиться, что ряды те же самые и что прогноз на GPU совпадает с прогнозом на CPU. Эталон CPU — пробный кеш `synth_v1_n1`: первый ряд каждой ячейки в нём тот же, что в полной сетке. Папку `tsfm_cache/synth_v1` скачать с Drive, например, в `~/Downloads/synth_v1`, и запустить:

```bash
python -m scripts.compare_caches results/cache/synth_v1 ~/Downloads/synth_v1 --model amazon/chronos-2      # ряды совпадают точно
python -m scripts.compare_caches results/cache/synth_v1_n1 ~/Downloads/synth_v1 --model amazon/chronos-2   # GPU против CPU
cp -r ~/Downloads/synth_v1/chronos-2 results/cache/synth_v1/      # только если обе сверки дали OK
```

Чтение кеша — `src/cache.py` (`load_windows`, `load_forecasts`).

## Этап 3: провалы прогноза (H1, H2)

Кривые ошибки и покрытия интервала в зависимости от d = t − τ, время восстановления d*, глобальное и условное покрытие. Нужен готовый кеш; исторический результат — `results/stage3/<модель>/`, протокол v1 сохраняется отдельно в `results/protocol_h1_h3_v1/stage3/<модель>/`:

```bash
python -m scripts.stage3_failures                              # chronos-bolt-small, ~30 с
python -m scripts.stage3_failures --model amazon/chronos-2
```

## Этап 4: сигналы надёжности (H3)

Сигналы S1–S5, S7 (`src/signals.py`) и их сравнение по Спирмену, AURC, AUROC и AUPRC (`src/metrics.py`). Исторические результаты — `results/stage4/<модель>/`; протокол v1 сохраняется в `results/protocol_h1_h3_v1/stage4/<модель>/`. S5 (эмбеддинги энкодера) кеш прогнозов не хранит, поэтому его считают отдельным прогоном энкодера, до этапа 4:

```bash
python -m scripts.cache_s5                       # chronos-bolt-small, ~18 мин на CPU (при двух параллельных расчётах) → results/cache/synth_v1/chronos-bolt-small/signals/s5.npz
python -m scripts.stage4_signals                 # chronos-bolt-small, ~30 мин на CPU (бутстреп)
python -m scripts.stage4_signals --n-boot 200    # быстрее, интервалы грубее
```

Проверка выводов на новых рядах той же сетки (другой seed); результат — `results/stage4/synth_v1_seed1/chronos-bolt-small/`:

```bash
python -m scripts.cache_forecasts --seed 1 --model amazon/chronos-bolt-small   # ~1 ч → results/cache/synth_v1_seed1/
python -m scripts.cache_s5 --cache results/cache/synth_v1_seed1
python -m scripts.stage4_signals --cache results/cache/synth_v1_seed1
```

## Этап B: локализация точки смены (H4)

Детекторы одной точки смены в контексте (`src/localization.py`) на каждом уровне энкодера и на сырых значениях. Эмбеддинги не сохраняются: детекторы считаются в том же проходе энкодера, на диск идут только счёт и τ′. Нужны оба seed: по seed 1 подбираются пороги и выбирается уровень, на seed 0 считается отчёт. Время — на GTX 1070 и 12 ядрах CPU, для каждого seed:

```bash
python -m scripts.cache_localization --model raw                                  # ~5 мин (PELT в 10 процессах)
python -m scripts.cache_localization --model amazon/chronos-bolt-small            # ~11 мин на CPU
python -m scripts.cache_localization --model amazon/chronos-2 --device cuda       # ~16 мин, оба горизонта
# то же с --cache results/cache/synth_v1_seed1
python -m scripts.stage5_localization                                             # таблицы и графики → results/stage5/
```

Пробный прогон на части ячеек: `--cells change-mean-strong,none --out /tmp/stage5_pilot`.

## Реальные данные: Petrobras 3W

Загрузка 1119 реальных экземпляров (516 МБ, ~1 ч, проверка git-blob SHA1) и описательный аудит. Протокол — `docs/protocol_3w_audit.md`, выводы — `docs/results/threew_audit.md`. Нужен `pyarrow`:

```bash
python -m scripts.download_3w             # → data/raw/3w/files/ (вне git), манифест data/raw/3w/manifest.json
python -m scripts.download_3w --check     # без сети: сверка файлов с манифестом
python -m scripts.audit_3w                # → results/threew_audit_v1/, ~1 мин
```

## Пилот Multi-Context (H7, план v3)

Выбор и смешивание прогнозов L64…L512 по готовому кешу, без инференса. Протокол — `docs/protocol_v3_multicontext.md`, выводы — `docs/results/adapt_multicontext.md`. Нужны кеши обоих seed, `cascade.npz`, `localization.npz`, `results/stage5/meta.json` и пороги `results/protocol_h1_h3_v1/stage3/`:

```bash
python -m scripts.adapt_multicontext                                          # Bolt H64 → results/adapt_multicontext_v1/chronos-bolt-small/H64/, ~11 мин CPU
python -m scripts.adapt_multicontext --horizon 16                             # ~3 мин
python -m scripts.adapt_multicontext --model amazon/chronos-2 --horizon 64    # ~11 мин
python -m scripts.adapt_multicontext --model amazon/chronos-2 --horizon 16
```

## Regime-Component (H9, план v3)

Без нового инференса: нужны основной кеш, `aligned_v1` (H8) и результаты H7/H8. Протокол — `docs/protocol_v3_component.md`, выводы — `docs/results/adapt_components.md`:

```bash
python -m scripts.adapt_components                                            # Bolt H64 → results/adapt_components_v1/…, ~20 с
python -m scripts.adapt_components --model amazon/chronos-2 --horizon 64
```

## Regime-Aligned Context (H8, план v3)

Новые прогнозы по выровненным контекстам (12 на окно), затем оценка по кешу. Протокол — `docs/protocol_v3_aligned.md`, выводы — `docs/results/adapt_aligned.md`:

```bash
python -m scripts.cache_aligned                                               # Bolt, CPU, ~27 мин → results/cache/synth_v1/chronos-bolt-small/aligned_v1/
python -m scripts.cache_aligned --model amazon/chronos-2 --device cuda        # ~39 мин на GTX 1070, оба горизонта
python -m scripts.adapt_aligned                                               # Bolt H64 → results/adapt_aligned_v1/…, ~25 с
python -m scripts.adapt_aligned --horizon 16
python -m scripts.adapt_aligned --model amazon/chronos-2 --horizon 64
```

## Каскад (H5)

Этап A (S4б или S2) отмечает 20% окон, этап B ищет в них τ′, прогноз пересчитывается по [τ′, t]; сравнение с постоянной детекцией, reset-and-grow и обрезкой по истинной τ. Нужны кеши обоих seed, этап B и `results/stage5/meta.json` (пороги). У Chronos-2 на seed 1 достаточно четырёх вариантов контекста:

```bash
python -m scripts.cache_forecasts --seed 1 --model amazon/chronos-2 --device cuda --variants L512 L256 L128 L64   # ~13 мин
python -m scripts.cache_cascade --model amazon/chronos-bolt-small            # ~4 мин на CPU → …/signals/cascade.npz
python -m scripts.cache_cascade --model amazon/chronos-2 --device cuda       # ~5 мин, оба горизонта
python -m scripts.stage6_cascade                                             # таблицы → results/stage6/chronos-bolt-small/, ~10 с
python -m scripts.stage6_cascade --model amazon/chronos-2
python -m scripts.stage6_cascade --signal s2                                 # этап A — S2 → results/stage6/s2/<модель>/
```
