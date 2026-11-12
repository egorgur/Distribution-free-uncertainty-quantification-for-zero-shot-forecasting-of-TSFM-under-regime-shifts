# PRONTO: уникальные дневные SCADA

Источник: https://zenodo.org/records/1341583, DOI10.5281/zenodo.1341583,
лицензия CC BY4.0. Получены только10членов `Process Data` по HTTP Range.
Сохранены четыре уникальных CSV; SHA256 всех повторов подтверждены по байтам,
CRC каждого ZIP-члена проверен. Источники/диапазоны/хеши — `manifest.json`.
Полный ZIP/видео/HF не скачаны; MD5 всего ZIP не проверен.

CSV игнорируются Git. Получение/сверка оригиналов:
`python -m scripts.download_pronto_scada`.
Аудит: `python -m scripts.audit_pronto_scada`, затем та же команда с `--check`.
Актуальные таблицы — `results/pronto_scada_v2/`, выводы —
`docs/results/pronto_daily_audit.md`.

Raw-время не исправлено:2017восновном минутные метки, хвост0911и2018 —
Excel serial. Совместимость с nominal1с не доказывает отсутствие внутриминутных
пропусков/дублей. Это не утверждённый набор/предобработка/эталон для H1–H3.