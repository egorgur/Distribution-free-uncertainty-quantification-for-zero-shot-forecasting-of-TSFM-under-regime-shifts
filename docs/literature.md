# Ключевая литература

**Область обзора на 2026-11-18:** список ниже обосновывает выполненную часть v2 — риск, надёжность, локализацию и каскад. Для [плана адаптации v3](adaptation_plan.md) нужен отдельный свежий обзор выбора/смешивания контекстов, коррекции истории, компонентной адаптации и последовательной калибровки. Он ещё не выполнен; новых проверенных источников этим обновлением не добавлено. Не переносить прежнюю формулировку пробела автоматически на четыре новых направления и не утверждать их уникальность.

Перед сравнением с reset-and-grow полностью прочитать «Breaking context inertia» и сверить с реализацией. Для conformal-ветки проверять предпосылки и вид гарантии, а не ссылаться на слово conformal как на доказательство условного покрытия при сменах.

Конспекты с цитатами — `papers/NOTES.md`, список PDF и что прочитано — `papers/README.md`.

Перенесено из `CLAUDE.md` 2026-11-01 без изменений по существу; обновлены только ссылки на разделы.

## Прямые конкуренты / ближайшие работы

- **Breaking context inertia: Adaptive context management for TSFMs in non-stationary environments** — Neurocomputing, 2026 (S0925231226015274). Детекция без обучения + reset-and-grow. **Главный конкурент этапа B / H5. Прочитать полностью.**
- **Shapelets-Enriched Selective Forecasting using TSFMs** — arXiv:2601.11821. Отказ по шейплетам; ошибки на валидации целевого домена; TTM. Заявленные −22% ошибки при отказе от 20% окон близки к тривиальному эффекту (ошибка отклонённого окна = 0); случайный отказ лучше на 2 датасетах из 6.
- **Selective Time Series Forecasting via Metalearning** — arXiv:2606.23448. Мета-модель (CatBoost по признакам TSFEL) предсказывает перцентиль ошибки; есть zero-shot перенос между доменами (без целевых меток, но с обучением на размеченном источнике); TSFM нет. Аналог и возможная реализация S6.
- **The Time-Energy Model** — TMLR 2025. Selective forecasting с обучением.

## Где и почему TSFM ломаются

- **Do TSFM Benchmarks Hide Regime-Dependent Failures?** — arXiv:2606.18367 (ICML 2026 workshop). **Ключевая.** Chronos-T5, Chronos-Bolt-Small, Moirai-1.1; METR-LA, PEMS-BAY; режим по скорости в горизонте; номинал 90%; длина контекста и время после смены не исследуются.
- **Causal Analysis for TSFMs** — arXiv:2608.24303.
- **Performance of Zero-Shot TSFMs on Cloud Data** — arXiv:2502.12944. Малое изменение контекста → сильно другой прогноз (опора S7).
- **Position: Universal TSFMs Rest on a Category Error** — arXiv:2602.05287.
- **TSFMs for Electricity Price Forecasting** — arXiv:2607.02623.

## Калибровка и неопределённость

- **Beyond Accuracy: Are TSFMs Well-Calibrated?** — Adler et al., ICLR 2026 (arXiv:2510.16060). Глобально хорошо откалиброваны (PCE ≲ 0.06 по квантилям 0.1–0.9 на 5 из 6 датасетов; chronos-bolt-base, TimesFM 2.0, Moirai 2.0 и др.); исключения — датасет Patents и авторегрессионные длинные горизонты (переуверенность). Влияние сдвига и нестационарности на калибровку прямо оставлено открытым. **Опора H2.**
- **TSFMs for Energy Load Forecasting on Consumer Hardware** — arXiv:2602.10848.
- **Evaluating Accuracy and Probabilistic Reliability of Zero-Shot TSFMs** — arXiv:2609.25788.
- **Quantifying Uncertainty in Foundation Models via Ensembles** — NeurIPS 2022 Workshop. Несогласованность как неопределённость (аналогия S4/S7).

## Контекст, смена режима, эмбеддинги

- **Why Do Time Series Models Need Long Context Windows?** — arXiv:2606.01999. **Мотивация S4** (не прямая опора): прогноз как байесовское усреднение по порождающим процессам, длинный контекст уточняет, какой процесс идёт. Теория только для стационарных процессов без смены; расхождение по длинам контекста не изучается. Следствие для нас: короткий контекст даёт разброс и без смены → обязателен контроль «без изменений».
- **In-Context Learning Under Regime Change** — arXiv:2604.16988 (ICML 2026). Знание точки смены улучшает прогноз.
- **Non-Stationarity in the Embedding Space of TSFMs** — Choi, Shook, Dubrawski, ICLR 2026 workshop (arXiv:2604.16428). Обнаруживаемость сдвигов (mean / variance / trend) в эмбеддингах Chronos-2, MOMENT, TOTEM линейным классификатором; у разных моделей разные провалы. Только синтетика AR(1), сдвиг всегда в середине окна, mean pooling выхода энкодера — **по слоям не анализирует**, с ошибкой прогноза не связывает. При сильных сдвигах простые статистики лучше эмбеддингов. **Опора S5 и пробела этапа B; для H4 по слоям — нет.**
- **Dissecting Chronos (SAE)** — arXiv:2603.10071. Признаки изменений в средних слоях Chronos-T5 (одна модель — не обобщать). Пока единственная опора H4 про слои.

## Аномалии и CPD

- **Exploring Zero-Shot FMs for MTS Anomaly Detection** — arXiv:2607.12454. TimesFM 1.2, w = 100, SWaT. Ошибка пикует на входе и выходе из аномального участка и на резких нормальных переходах (ложные тревоги); CPD — «будущая работа». **Ключевая.**
- **THEMIS** — arXiv:2510.03911; **ChronosAD** — arXiv:2606.01300; **Adaptive Conformal AD with TSFMs** (IBM) — arXiv:2604.20122; **StarEmbed** — arXiv:2510.06200.
- **TS-CP²** — arXiv:2011.14097.

## OOD для временных рядов

- **TS-OOD** — arXiv:2502.15901; **Hyperspherical TF Representations for TS OOD** — arXiv:2605.31155 (IJCAI 2026); **CODiT** — arXiv:2207.11769.

## Прочее

- **Anytime-Valid Gating on Real Forecast Streams** — arXiv:2608.30502.
- **Post-Training in TSFMs: A Unifying Framework** — arXiv:2607.20002.
- **Do TSFMs Pay Off for Industrial Monitoring?** — arXiv:2608.22968.
- **FETS Benchmark** — arXiv:2604.22328.
- **Chronos** — arXiv:2403.07815.

## Бенчмарки

- **TSSB** (Time Series Segmentation Benchmark), **TCPD** (Turing Change Point Dataset), **GIFT-Eval**.

> Многие работы просмотрены только по аннотациям. Перед цитированием — прочитать полностью и сверить формулировки.
