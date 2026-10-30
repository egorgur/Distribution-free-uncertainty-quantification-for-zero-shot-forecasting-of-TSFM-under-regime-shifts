# Ключевые статьи

Список «Полностью прочитать» из `CLAUDE.md` (раздел 11). PDF скачаны с arXiv 2026-10-30, ID и названия сверены через arXiv API.
Сами PDF не хранятся в git (см. `.gitignore`): у статей разные лицензии. Чтобы скачать заново, используйте ссылки ниже.
Конспекты релевантных частей с цитатами — в [`NOTES.md`](NOTES.md).

| Файл | Статья | Зачем | Прочитано |
| --- | --- | --- | --- |
| — (нет открытого PDF) | Breaking context inertia: Adaptive context management for TSFMs in non-stationary environments. Neurocomputing, 2026. [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0925231226015274) | Главный конкурент этапа B / H5 | [ ] |
| `2601.11821_Shapelets_Selective_Forecasting.pdf` | Shapelets-Enriched Selective Forecasting using Time Series Foundation Models. [arXiv:2601.11821](https://arxiv.org/abs/2601.11821) | Отказ от прогноза с разметкой (пробел) | [ ] |
| `2606.23448_Selective_Forecasting_Metalearning.pdf` | Selective Time Series Forecasting via Metalearning. [arXiv:2606.23448](https://arxiv.org/abs/2606.23448) | Аналог S6 | [ ] |
| `2606.18367_TSFM_Benchmarks_Regime_Failures.pdf` | Do Time Series Foundation Model Benchmarks Hide Regime-Dependent Failures? Evidence from Traffic Speed Forecasting. [arXiv:2606.18367](https://arxiv.org/abs/2606.18367) | Провалы в переходах (H1, H2) | [ ] |
| `2510.16060_Adler_TSFM_Calibration_ICLR2026.pdf` | Adler et al. Beyond Accuracy: Are Time Series Foundation Models Well-Calibrated? ICLR 2026. [arXiv:2510.16060](https://arxiv.org/abs/2510.16060) | Глобальная калибровка (H2) | [ ] |
| `2604.16428_Choi_NonStationarity_Embedding_Space.pdf` | Choi, Shook, Dubrawski. Non-Stationarity in the Embedding Space of Time Series Foundation Models. ICLR 2026 TSALM Workshop. [arXiv:2604.16428](https://arxiv.org/abs/2604.16428) | Сдвиги в эмбеддингах (S5; по слоям не анализирует) | [ ] |
| `2606.01999_Why_Long_Context_Windows.pdf` | Butera et al. Why Do Time Series Models Need Long Context Windows? [arXiv:2606.01999](https://arxiv.org/abs/2606.01999) | Теоретическая мотивация S4 (только стационарный случай) | [ ] |
| `2607.12454_ZeroShot_FM_MTS_Anomaly_Detection.pdf` | Exploring Zero-Shot Foundation Models for Multivariate Time Series Anomaly Detection. [arXiv:2607.12454](https://arxiv.org/abs/2607.12454) | Ошибка пикует на границах режимов | [ ] |

Скачать всё заново:

```bash
cd papers
for id in 2601.11821 2606.23448 2606.18367 2510.16060 2604.16428 2606.01999 2607.12454; do
  curl -sSL -o "$id.pdf" "https://arxiv.org/pdf/$id"; sleep 3
done
```
