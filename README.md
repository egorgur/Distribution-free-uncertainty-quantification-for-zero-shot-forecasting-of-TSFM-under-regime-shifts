# Distribution-free-uncertainty-quantification-for-zero-shot-forecasting-of-TSFM-under-regime-shifts

Модели вроде Chronos или TimesFM прогнозируют временные ряды без дообучения, но не умеют сообщать, когда их прогнозу нельзя верить. Хуже всего они работают в моменты смены режима процесса: при переходе от одного состояния к другому ошибка резко растёт, а прогнозные интервалы перестают покрывать реальные значения.

## Установка

Локально, на CPU (нужен [uv](https://docs.astral.sh/uv/)):

```bash
uv venv .venv --python 3.13
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python -r requirements.txt
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

## Синтетика и кеш прогнозов

Сетка факторов задана в `configs/synth_v1.yaml`. Примеры рядов каждого типа сохраняются в `results/synth_v1/examples.png`:

```bash
python -m scripts.plot_synth
```

Прогнозы всех окон и всех вариантов контекста (S4, S7) сохраняются в `results/cache/synth_v1/` (в git не попадает). Если прогон прервался, та же команда продолжит его с места остановки:

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

Кривые ошибки и покрытия интервала в зависимости от d = t − τ, время восстановления d*, глобальное и условное покрытие. Нужен готовый кеш; результат — `results/stage3/<модель>/` (таблицы в `summary.md`, графики `fig_*.png`):

```bash
python -m scripts.stage3_failures                              # chronos-bolt-small, ~30 с
python -m scripts.stage3_failures --model amazon/chronos-2
```

## Этап 4: сигналы надёжности (H3)

Сигналы S1–S5, S7 (`src/signals.py`) и их сравнение по Спирмену, AURC, AUROC и AUPRC (`src/metrics.py`). Нужен готовый кеш; результат — `results/stage4/<модель>/`. S5 (эмбеддинги энкодера) кеш прогнозов не хранит, поэтому его считают отдельным прогоном энкодера, до этапа 4:

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
