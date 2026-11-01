# Сделанное

Выполненные пункты по порядку, с временем расчётов и результатами сверок. Открытые пункты — `CLAUDE.md`, раздел 11.

Перенесено из `CLAUDE.md` 2026-11-01 без изменений по существу; обновлены только ссылки на разделы.

- [x] Выбрана и уточнена тема, проведён первичный обзор литературы (октябрь 2026)
- [x] Сформулированы задача, гипотезы и план (v1)
- [x] Получены замечания научрука (11 пунктов + статья про context inertia), постановка обновлена до v2 (`TSFM_regime_reliability_plan_v2.docx`)
- [x] Скачаны PDF ключевых статей в `papers/` (кроме «Breaking context inertia» — нет открытого доступа); список в `papers/README.md`
- [x] Агентом прочитаны релевантные части 7 скачанных статей; конспекты с цитатами — `papers/NOTES.md`; по итогам уточнены пробел и описания литературы
- [x] Этап 0, часть 1: локальное окружение (`.venv`, `requirements.txt`), обёртка Chronos-Bolt (`src/models.py`), проверка `scripts/check_chronos.py` — все проверки пройдены (2026-10-30)
- [x] Сравнены 6 моделей Chronos (Bolt tiny…base, Chronos-2 small и полная) на синтетике без смены: `results/model_comparison/`, выводы — `docs/results/model_comparison.md` (2026-10-30)
- [x] Вторая модель для H6 — Chronos-2 (`amazon/chronos-2`) (2026-10-30)
- [x] Этап 1: генератор синтетики `src/synth.py`, сетка `configs/synth_v1.yaml`, окна и варианты S4/S7 `src/windows.py` (2026-10-30)
- [x] Этап 0, часть 2: кеш прогнозов `scripts/cache_forecasts.py`, чтение `src/cache.py`; пробный прогон обеих моделей прошёл (2026-10-30)
- [x] Полный кеш bolt-small: `results/cache/synth_v1/`, 55 мин на CPU, 731 МБ (2026-10-30)
- [x] Полный кеш chronos-2: `results/cache/synth_v1/chronos-2/`, 32.5 мин на GPU GTX 1070 (второй компьютер), 907 МБ; пересчёт и сверка с эталоном CPU прошли — `docs/data.md`, «Кеш прогнозов» (2026-11-01)
- [x] Этап 3 (H1, H2) на chronos-bolt-small: `results/stage3/chronos-bolt-small/`, выводы — `docs/results/stage3_failures.md` (2026-10-30)
- [x] Этап 3 на chronos-2: `results/stage3/chronos-2/`, выводы — `docs/results/stage3_failures.md` (2026-11-01)
- [x] Этап 4 (H3) на chronos-bolt-small: S1–S4, S7 по кешу, `results/stage4/chronos-bolt-small/`, выводы — `docs/results/stage4_signals.md` (2026-11-01)
- [x] Сочетания сигналов посчитаны (полный прогон, 1000 повторов), выводы — `docs/results/stage4_signals.md`, «Сочетания сигналов» (2026-11-01). Полный прогон `scripts.stage4_signals` занимает ~28 мин
- [x] Проверка сочетаний на seed 1 (критерий — `docs/decisions.md`): критерий выполнен, +0.027 [0.011; 0.042]; выводы — `docs/results/stage4_signals.md`, «Проверка сочетаний на seed 1» (2026-11-01). Кеш `results/cache/synth_v1_seed1/` полный: 18 ячеек посчитаны на ноутбуке (torch 2.15.0+cpu), 16 — на втором компьютере (torch 2.14.1+cu126, на CPU); поле `env` в `meta.json` — второго компьютера. Сверка 8 случайных окон (6 из них — из ячеек ноутбука): 1.7e-6 std контекста. S5 — 125 с, этап 4 — 9.3 мин
- [x] Закоммичены код, документы и результаты этапов 1–4 (2026-11-01)
- [x] S5: код (`encode_patches` в `src/models.py`, `embedding_shift` в `src/signals.py`, `scripts/cache_s5.py`), посчитан на synth_v1 → `results/cache/synth_v1/chronos-bolt-small/signals/s5.npz`; определение — `docs/implementation.md` и `docs/decisions.md` (2026-11-01)
- [x] Этап 4 с S5 на synth_v1: `results/stage4/chronos-bolt-small/`, выводы — `docs/results/stage4_signals.md`, «S5: сдвиг эмбеддингов» (2026-11-01, второй компьютер, 9.5 мин на CPU). `git diff`: в `summary.md` изменились только строки S5, дата и строка об S5 в «Ограничениях»; в CSV прежние строки и столбцы совпадают, добавлены только S5
- [x] Кеш прогнозов добавлен в git (2026-11-01): `results/cache/` — synth_v1 (bolt-small целиком + S5), synth_v1_n1 (пробный, эталон CPU для `scripts.compare_caches`), synth_v1_seed1 (18 из 34 ячеек). На другом компьютере достаточно `git clone` и установки окружения (`README.md`). Память агента и история чата не переносятся: всё нужное — в `CLAUDE.md` и `docs/`
- [x] S5 для Chronos-2: `encode_patches` в `Chronos2Forecaster`, S5 отдельно для каждого горизонта; определение записано в журнал решений до расчёта; 253 с на GPU → `results/cache/synth_v1/chronos-2/signals/s5.npz`; сверки прошли — `docs/implementation.md` (2026-11-01)
- [x] Этап 4 на chronos-2: `results/stage4/chronos-2/`, 9.5 мин на CPU; выводы — `docs/results/stage4_signals.md`, «на chronos-2» (2026-11-01)
- [x] Этап B (H4) на синтетике: план и решения записаны в журнал до расчёта (2026-11-01)
- [x] Этап B (H4): код, пробный прогон, расчёт на seed 1 (подбор) и seed 0 (отчёт) → `results/stage5/`; критерий выполнен на Chronos-2 (блок 8, +0.04 [0.01; 0.07]) и не выполнен на Bolt; выводы — `docs/results/stage5_localization.md` (2026-11-01). Не закоммичено
- [x] Каскад (H5): решения, политики и критерий записаны в журнал до расчёта, в том числе второй прогон с S2 (2026-11-02)
- [x] Кеш Chronos-2 на seed 1, только варианты L512, L256, L128, L64 (новая опция `cache_forecasts --variants`): 263 840 прогнозов, 13 мин на GPU (3.0 мс на прогноз), 392 МБ; пересчёт 8 окон — 6.1e-6 std контекста. Лог — `results/cache/synth_v1_seed1_chronos-2_gpu.log` (2026-11-02)
- [x] Прогнозы по обрезанным контекстам на seed 0 (`scripts/cache_cascade.py` → `cascade.npz`, 7 МБ на модель): Bolt — 83 133 прогноза, 256 с на CPU (3.0 мс); Chronos-2 — 162 144 прогноза за оба H, 300 с на GPU (1.6–1.8 мс). Сверки: полный контекст в общем батче против кеша — 2.0e-6 и 2.2e-6 std контекста; обрезанный контекст в одиночку — 1.2e-6 и 2.5e-6; независимый пересчёт 32 обрезок Bolt всех видов — MAE совпал до 4.7e-7 (2026-11-02)
- [x] Каскад (H5) на синтетике, обе модели, этап A — S4б, затем S2 → `results/stage6/`, `results/stage6/s2/`; `scripts.stage6_cascade` — 4–11 с. Критерий не выполнен (К1 1.049 [1.031; 1.070], К2 −21.7 п. п.); выводы — `docs/results/stage6_cascade.md` (2026-11-02). Не закоммичено
- [x] Отчёт научному руководителю: что требовалось (план v2, 11 замечаний), что сделано, итоги H1–H6, отклонения от плана, вопросы для решения → `docs/supervisor_report_2026-11-02.md`, Word-версия `TSFM_progress_report_2026-11-02.docx` (pandoc). Цифры сверены с `docs/results/` (2026-11-02)
