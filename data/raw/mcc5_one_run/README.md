# MCC5-THU: один исходный запуск

Данные Chen et al., *Multi-mode fault diagnosis datasets of three-phase
asynchronous motor under variable working conditions*, DOI10.1016/j.dib.2026.112583.
Источник поставки — зеркало, связанное карточкой Mendeley V2:
https://data.mendeley.com/datasets/6s3dggj9mw/2 (CC BY4.0).
https://huggingface.co/datasets/Samlzy/MCC5-THU-Motor
commit `897bc5d2504980418249f98e733bfd7f61056186`.

README зеркала заявляет MIT; тождество его лицензии/чисел с оригинальным
Mendeley не установлено. Не смешивать лицензии и не распространять CSV
под предположенной лицензией без уточнения. При использовании цитировать авторов.

`selected.csv` — оригинальные118 663 524байта одного healthy torque20Nm/1000rpm
запуска, исключены изGit. Точный архивный путь, CRC и SHA256 — `manifest.json`.
Полный6.51GB ZIP не скачан, его LFS SHA256 не проверен.
Сохранены pinned `source_README.md`, `hf_tree.json`, ZIP `inventory.json`,
`selection.json`, происхождение HTTP-диапазонов и `source_article.xml`.
Размер/счётчик загрузки относятся к успешному проходу; ранее был TLS-тайм-аут.

Из корня: `python -m scripts.download_mcc5 --inventory-only` сначала сохраняет
оглавление и выбор. Перед последующим `python -m scripts.download_mcc5` точный
путь нужно записать в журнал до численного чтения. Локальная сверка —
`python -m scripts.download_mcc5 --check`. Ответ200 на Range отклоняется
до чтения тела, установлены лимиты размера/передачи.

Новый аудит требует пустой каталог, например `python -m scripts.audit_mcc5 --out results/mcc5_repeat`.
Проверка актуального снимка: `python -m scripts.audit_mcc5 --out results/mcc5_one_run_v2 --check`.
Первый v1 — исторический после уточнения типов кода, результаты численно
и побайтово не изменились. Не перезаписывать его хеши.

Схема9столбцов: время +8сигналов, заголовка нет. Числа не конвертировались,
масштаб момента не уточнён. Отчёт — `docs/results/mcc5_one_run_audit.md`.
Стабильность/эталон/прогнозный протокол не утверждены, TSFM не запускались.