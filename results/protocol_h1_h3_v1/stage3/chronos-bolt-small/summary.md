# Этап 3 — amazon/chronos-bolt-small, протокол v1

Дата: 2026-11-12 21:28. Протокол: `h1_h3_v1`, SHA256 `22bae05a71811c6cf7c17b2bac1d3a80ff106f010f2eff3cabd612bb816e8dc1`.
Оценка: `results/cache/synth_v1`, seed 0; контроль: `results/cache/synth_v1_seed1`, seed 1.

## Определения и ограничения

- MAE / медианная MAE того же ряда/model/L/H. Старый эталон d < −H; новый 512 ≤ d ≤ 640.
- В горизонте p = clip((d+H+1)/H,0,1), смесь (1−p)b_old+p b_new. У none/anomaly всегда старый эталон.
- Drift: полоса 512–640 — приближение, не строго устойчивый контекст; ниже есть анализ без drift.
- Геометрическое среднее по 10 рядам. Для log при нулевом отношении floor 1e-06; невалидные знаменатели исключаются, не заменяются epsilon. Сами отношения не обрезаются.
- Контроль и оценка: log-сглаживание по 3 соседям строго внутри 0,8,…,504. Q95 основной, Q90 чувствительность; линейный процентиль, threshold=max(1,Q). Нет выбора по seed0, типу или силе.
- d* строго после последнего максимума сглаженной MAE: 9 значений ≤ порога (span64), на остатке ≤floor(n/10) превышений и максимум два подряд.
- no_degradation: превышений нет, численное d* не назначается. recovered: найдено d*. not_confirmed: подтверждения нет. Поиск до 504, последний кандидат 440. not_confirmed НЕ означает d* ≥ 512.
- Бутстреп 1000 раз по рядам целиком. CI d* условен только на recovered-повторах и фиксированном контроле seed1, неопределённость контроля в него не входит. Частоты всех статусов приведены рядом.
- CRPS, ширина и MASE — описательные кривые, d* для CRPS с порогами MAE не вычисляется.
- Обе синтетические выборки уже изучались; пересчёт не является новым независимым подтверждением H1.

## Фиксированные пороги (отдельно по L/H)

| L | H | Q | Сырой Q | Порог |
| --- | --- | --- | --- | --- |
| 64 | 16 | 95 | 1.055391 | 1.055391 |
| 64 | 16 | 90 | 1.037640 | 1.037640 |
| 64 | 64 | 95 | 1.026364 | 1.026364 |
| 64 | 64 | 90 | 0.988097 | 1.000000 |
| 128 | 16 | 95 | 1.113912 | 1.113912 |
| 128 | 16 | 90 | 1.101851 | 1.101851 |
| 128 | 64 | 95 | 1.031077 | 1.031077 |
| 128 | 64 | 90 | 1.014082 | 1.014082 |
| 256 | 16 | 95 | 1.077159 | 1.077159 |
| 256 | 16 | 90 | 1.062542 | 1.062542 |
| 256 | 64 | 95 | 1.090462 | 1.090462 |
| 256 | 64 | 90 | 1.051765 | 1.051765 |
| 512 | 16 | 95 | 1.089373 | 1.089373 |
| 512 | 16 | 90 | 1.072328 | 1.072328 |
| 512 | 64 | 95 | 1.046197 | 1.046197 |
| 512 | 64 | 90 | 1.023873 | 1.023873 |

## H1: H=16, Q95 (основной)

CI в таблице — только по recovered-повторам. Drift помечен `≈`.

| Ячейка | L | Сглаженный пик @ d | Статус | d* | d*/L | Условный CI95 d* | boot: нет дегр. | boot: recovered | boot: не подтв. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| change-mean-weak | 512 | 1.18 @ 216 | not_confirmed | — | — | [142.6; 432] | 0 | 274 | 726 |
| change-mean-medium | 512 | 1.31 @ 0 | recovered | 440 | 0.859 | [296; 440] | 0 | 723 | 277 |
| change-mean-strong | 512 | 1.70 @ 0 | not_confirmed | — | — | [56; 434.4] | 0 | 229 | 771 |
| change-variance-weak | 512 | 1.21 @ 280 | not_confirmed | — | — | [48; 304] | 3 | 124 | 873 |
| change-variance-medium | 512 | 1.19 @ 264 | recovered | 336 | 0.656 | [272; 440] | 0 | 470 | 530 |
| change-variance-strong | 512 | 1.17 @ 504 | not_confirmed | — | — | [88; 328] | 1 | 492 | 507 |
| change-trend-weak | 512 | 1.26 @ 448 | not_confirmed | — | — | — | 0 | 0 | 1000 |
| change-trend-medium | 512 | 1.25 @ 224 | recovered | 320 | 0.625 | [104; 350.2] | 0 | 610 | 390 |
| change-trend-strong | 512 | 1.38 @ 72 | recovered | 232 | 0.453 | [144; 424] | 0 | 604 | 396 |
| change-ar-weak | 512 | 1.18 @ 32 | recovered | 208 | 0.406 | [88; 368] | 0 | 631 | 369 |
| change-ar-medium | 512 | 1.14 @ 312 | recovered | 328 | 0.641 | [112; 440] | 6 | 575 | 419 |
| change-ar-strong | 512 | 1.11 @ 24 | recovered | 32 | 0.062 | [32; 344] | 54 | 651 | 295 |
| change-seasonality-weak | 512 | 1.37 @ 8 | recovered | 40 | 0.078 | [32; 408] | 0 | 509 | 491 |
| change-seasonality-medium | 512 | 1.33 @ 0 | recovered | 96 | 0.188 | [40; 392] | 0 | 596 | 404 |
| change-seasonality-strong | 512 | 1.63 @ 0 | recovered | 200 | 0.391 | [32; 344] | 0 | 947 | 53 |
| change-frequency-weak | 512 | 1.46 @ 144 | not_confirmed | — | — | [280; 432] | 0 | 231 | 769 |
| change-frequency-medium | 512 | 1.78 @ 56 | not_confirmed | — | — | [282.8; 432] | 0 | 15 | 985 |
| change-frequency-strong | 512 | 2.19 @ 56 | recovered | 368 | 0.719 | [264; 408] | 0 | 728 | 272 |
| change-sensor_bias-weak | 512 | 1.12 @ 168 | recovered | 176 | 0.344 | [64; 400] | 14 | 683 | 303 |
| change-sensor_bias-medium | 512 | 1.18 @ 0 | not_confirmed | — | — | [48; 432] | 0 | 152 | 848 |
| change-sensor_bias-strong | 512 | 2.00 @ 0 | not_confirmed | — | — | [360; 424] | 0 | 171 | 829 |
| drift-mean-weak ≈ | 512 | 1.22 @ 368 | recovered | 392 | 0.766 | [368; 440] | 0 | 451 | 549 |
| drift-mean-medium ≈ | 512 | 1.34 @ 48 | recovered | 280 | 0.547 | [88; 432] | 0 | 625 | 375 |
| drift-mean-strong ≈ | 512 | 1.29 @ 88 | recovered | 360 | 0.703 | [144; 408] | 0 | 792 | 208 |
| drift-frequency-weak ≈ | 512 | 1.40 @ 264 | recovered | 352 | 0.688 | [312; 440] | 0 | 764 | 236 |
| drift-frequency-medium ≈ | 512 | 1.49 @ 128 | not_confirmed | — | — | [328; 368] | 0 | 23 | 977 |
| drift-frequency-strong ≈ | 512 | 1.78 @ 176 | not_confirmed | — | — | [336; 424] | 0 | 366 | 634 |
| anomaly-spike-weak | 512 | 1.25 @ 184 | recovered | 416 | 0.812 | [200; 433] | 0 | 476 | 524 |
| anomaly-spike-medium | 512 | 1.38 @ 0 | recovered | 120 | 0.234 | [16; 440] | 1 | 737 | 262 |
| anomaly-spike-strong | 512 | 2.11 @ 0 | recovered | 416 | 0.812 | [200; 440] | 0 | 496 | 504 |
| anomaly-freeze-weak | 512 | 1.13 @ 0 | recovered | 8 | 0.016 | [8; 376] | 72 | 407 | 521 |
| anomaly-freeze-medium | 512 | 1.97 @ 0 | not_confirmed | — | — | [296; 440] | 0 | 86 | 914 |
| anomaly-freeze-strong | 512 | 1.57 @ 0 | not_confirmed | — | — | [8; 426.4] | 0 | 149 | 851 |
| none | 512 | 1.15 @ 80 | recovered | 136 | 0.266 | [80; 416] | 77 | 480 | 443 |
| change-mean-weak | 256 | 1.29 @ 216 | not_confirmed | — | — | [253.6; 440] | 0 | 77 | 923 |
| change-mean-medium | 256 | 1.23 @ 280 | recovered | 304 | 1.188 | [16; 440] | 0 | 840 | 160 |
| change-mean-strong | 256 | 1.63 @ 0 | not_confirmed | — | — | [64; 416] | 0 | 289 | 711 |
| change-variance-weak | 256 | 1.20 @ 280 | not_confirmed | — | — | [288; 383.6] | 0 | 203 | 797 |
| change-variance-medium | 256 | 1.21 @ 416 | not_confirmed | — | — | [280; 440] | 0 | 123 | 877 |
| change-variance-strong | 256 | 1.16 @ 504 | not_confirmed | — | — | [101.2; 400] | 0 | 507 | 493 |
| change-trend-weak | 256 | 1.29 @ 448 | not_confirmed | — | — | — | 0 | 0 | 1000 |
| change-trend-medium | 256 | 1.16 @ 216 | recovered | 360 | 1.406 | [96; 376] | 0 | 610 | 390 |
| change-trend-strong | 256 | 1.25 @ 72 | recovered | 256 | 1.000 | [93.4; 424] | 0 | 468 | 532 |
| change-ar-weak | 256 | 1.16 @ 344 | not_confirmed | — | — | [208; 376] | 0 | 209 | 791 |
| change-ar-medium | 256 | 1.13 @ 0 | not_confirmed | — | — | [32; 336] | 10 | 405 | 585 |
| change-ar-strong | 256 | 1.13 @ 24 | recovered | 32 | 0.125 | [32; 344] | 0 | 719 | 281 |
| change-seasonality-weak | 256 | 1.44 @ 16 | recovered | 112 | 0.438 | [40; 424] | 0 | 656 | 344 |
| change-seasonality-medium | 256 | 1.44 @ 0 | recovered | 96 | 0.375 | [40; 432] | 0 | 600 | 400 |
| change-seasonality-strong | 256 | 1.57 @ 0 | recovered | 88 | 0.344 | [32; 256] | 0 | 959 | 41 |
| change-frequency-weak | 256 | 1.23 @ 144 | not_confirmed | — | — | [170.2; 432] | 0 | 92 | 908 |
| change-frequency-medium | 256 | 1.54 @ 56 | not_confirmed | — | — | [64; 363.2] | 0 | 109 | 891 |
| change-frequency-strong | 256 | 1.99 @ 56 | recovered | 232 | 0.906 | [168; 408] | 0 | 395 | 605 |
| change-sensor_bias-weak | 256 | 1.13 @ 168 | recovered | 184 | 0.719 | [56; 432] | 43 | 775 | 182 |
| change-sensor_bias-medium | 256 | 1.20 @ 0 | not_confirmed | — | — | [8; 424] | 0 | 46 | 954 |
| change-sensor_bias-strong | 256 | 1.79 @ 0 | not_confirmed | — | — | [216; 416] | 0 | 201 | 799 |
| drift-mean-weak ≈ | 256 | 1.24 @ 328 | recovered | 400 | 1.562 | [368; 440] | 0 | 487 | 513 |
| drift-mean-medium ≈ | 256 | 1.34 @ 48 | not_confirmed | — | — | [147; 432] | 0 | 332 | 668 |
| drift-mean-strong ≈ | 256 | 1.32 @ 88 | recovered | 400 | 1.562 | [272; 432] | 0 | 558 | 442 |
| drift-frequency-weak ≈ | 256 | 1.14 @ 144 | recovered | 176 | 0.688 | [152; 336] | 4 | 436 | 560 |
| drift-frequency-medium ≈ | 256 | 1.29 @ 120 | not_confirmed | — | — | [136; 184] | 0 | 290 | 710 |
| drift-frequency-strong ≈ | 256 | 1.57 @ 72 | not_confirmed | — | — | [184; 440] | 0 | 188 | 812 |
| anomaly-spike-weak | 256 | 1.17 @ 264 | recovered | 280 | 1.094 | [32; 416] | 5 | 573 | 422 |
| anomaly-spike-medium | 256 | 1.50 @ 0 | recovered | 128 | 0.500 | [16; 440] | 0 | 422 | 578 |
| anomaly-spike-strong | 256 | 2.18 @ 0 | not_confirmed | — | — | [368; 440] | 0 | 308 | 692 |
| anomaly-freeze-weak | 256 | 1.08 @ 0 | recovered | 8 | 0.031 | [8; 392] | 38 | 506 | 456 |
| anomaly-freeze-medium | 256 | 1.64 @ 0 | not_confirmed | — | — | [224; 440] | 0 | 31 | 969 |
| anomaly-freeze-strong | 256 | 1.63 @ 0 | not_confirmed | — | — | [336; 432] | 0 | 74 | 926 |
| none | 256 | 1.20 @ 80 | recovered | 168 | 0.656 | [88; 424] | 8 | 670 | 322 |
| change-mean-weak | 128 | 1.29 @ 216 | not_confirmed | — | — | [240; 440] | 0 | 248 | 752 |
| change-mean-medium | 128 | 1.22 @ 288 | recovered | 304 | 2.375 | [40; 440] | 0 | 798 | 202 |
| change-mean-strong | 128 | 1.33 @ 0 | recovered | 144 | 1.125 | [16; 440] | 0 | 667 | 333 |
| change-variance-weak | 128 | 1.20 @ 280 | recovered | 304 | 2.375 | [56; 376] | 24 | 616 | 360 |
| change-variance-medium | 128 | 1.19 @ 264 | recovered | 288 | 2.250 | [113; 432] | 0 | 846 | 154 |
| change-variance-strong | 128 | 1.17 @ 208 | recovered | 224 | 1.750 | [104; 422.4] | 87 | 689 | 224 |
| change-trend-weak | 128 | 1.27 @ 216 | not_confirmed | — | — | — | 0 | 0 | 1000 |
| change-trend-medium | 128 | 1.32 @ 224 | recovered | 272 | 2.125 | [96; 408] | 0 | 571 | 429 |
| change-trend-strong | 128 | 1.30 @ 368 | not_confirmed | — | — | [320; 429.4] | 0 | 254 | 746 |
| change-ar-weak | 128 | 1.22 @ 0 | not_confirmed | — | — | [56; 368] | 0 | 288 | 712 |
| change-ar-medium | 128 | 1.26 @ 8 | not_confirmed | — | — | [16; 440] | 3 | 376 | 621 |
| change-ar-strong | 128 | 1.16 @ 16 | recovered | 32 | 0.250 | [24; 272] | 26 | 852 | 122 |
| change-seasonality-weak | 128 | 1.51 @ 16 | recovered | 40 | 0.312 | [32; 424] | 0 | 796 | 204 |
| change-seasonality-medium | 128 | 1.45 @ 0 | recovered | 96 | 0.750 | [24; 376] | 0 | 623 | 377 |
| change-seasonality-strong | 128 | 1.53 @ 0 | recovered | 88 | 0.688 | [32; 264] | 0 | 884 | 116 |
| change-frequency-weak | 128 | 1.14 @ 168 | recovered | 184 | 1.438 | [152; 424] | 23 | 308 | 669 |
| change-frequency-medium | 128 | 1.27 @ 56 | not_confirmed | — | — | [64; 248] | 0 | 268 | 732 |
| change-frequency-strong | 128 | 1.44 @ 16 | recovered | 104 | 0.812 | [64; 384] | 0 | 887 | 113 |
| change-sensor_bias-weak | 128 | 1.23 @ 176 | recovered | 184 | 1.438 | [88; 408] | 5 | 912 | 83 |
| change-sensor_bias-medium | 128 | 1.20 @ 88 | recovered | 96 | 0.750 | [8; 416] | 6 | 390 | 604 |
| change-sensor_bias-strong | 128 | 1.63 @ 0 | not_confirmed | — | — | [112; 440] | 0 | 123 | 877 |
| drift-mean-weak ≈ | 128 | 1.27 @ 368 | recovered | 392 | 3.062 | [376; 440] | 0 | 715 | 285 |
| drift-mean-medium ≈ | 128 | 1.19 @ 80 | recovered | 168 | 1.312 | [64; 440] | 0 | 656 | 344 |
| drift-mean-strong ≈ | 128 | 1.19 @ 344 | recovered | 368 | 2.875 | [88; 440] | 0 | 861 | 139 |
| drift-frequency-weak ≈ | 128 | 1.17 @ 152 | recovered | 168 | 1.312 | [160; 440] | 2 | 524 | 474 |
| drift-frequency-medium ≈ | 128 | 1.20 @ 456 | not_confirmed | — | — | [136; 376] | 7 | 79 | 914 |
| drift-frequency-strong ≈ | 128 | 1.16 @ 48 | recovered | 88 | 0.688 | [56; 440] | 111 | 667 | 222 |
| anomaly-spike-weak | 128 | 1.42 @ 0 | recovered | 32 | 0.250 | [16; 432] | 0 | 602 | 398 |
| anomaly-spike-medium | 128 | 1.71 @ 0 | not_confirmed | — | — | [16; 392.4] | 0 | 399 | 601 |
| anomaly-spike-strong | 128 | 3.03 @ 0 | not_confirmed | — | — | [208; 440] | 0 | 343 | 657 |
| anomaly-freeze-weak | 128 | 1.18 @ 264 | recovered | 272 | 2.125 | [8; 392] | 16 | 521 | 463 |
| anomaly-freeze-medium | 128 | 1.22 @ 56 | not_confirmed | — | — | [89.2; 440] | 0 | 382 | 618 |
| anomaly-freeze-strong | 128 | 2.07 @ 136 | not_confirmed | — | — | [176; 440] | 0 | 282 | 718 |
| none | 128 | 1.08 @ 120 | no_degradation | — | — | [128; 424] | 204 | 649 | 147 |
| change-mean-weak | 64 | 1.27 @ 232 | not_confirmed | — | — | [288; 432] | 0 | 97 | 903 |
| change-mean-medium | 64 | 1.21 @ 288 | not_confirmed | — | — | [300.6; 440] | 0 | 224 | 776 |
| change-mean-strong | 64 | 1.35 @ 0 | not_confirmed | — | — | [84.8; 432] | 0 | 265 | 735 |
| change-variance-weak | 64 | 1.25 @ 40 | not_confirmed | — | — | [48; 437.6] | 4 | 133 | 863 |
| change-variance-medium | 64 | 1.18 @ 48 | recovered | 432 | 6.750 | [280; 432] | 0 | 564 | 436 |
| change-variance-strong | 64 | 1.17 @ 216 | recovered | 240 | 3.750 | [109; 432] | 14 | 666 | 320 |
| change-trend-weak | 64 | 1.33 @ 384 | not_confirmed | — | — | [400; 400] | 0 | 4 | 996 |
| change-trend-medium | 64 | 1.20 @ 48 | not_confirmed | — | — | [376; 439.2] | 0 | 85 | 915 |
| change-trend-strong | 64 | 1.10 @ 224 | recovered | 232 | 3.625 | [104; 424] | 0 | 769 | 231 |
| change-ar-weak | 64 | 1.24 @ 504 | not_confirmed | — | — | [312; 312] | 0 | 1 | 999 |
| change-ar-medium | 64 | 1.26 @ 0 | not_confirmed | — | — | [120; 440] | 0 | 74 | 926 |
| change-ar-strong | 64 | 1.12 @ 16 | recovered | 136 | 2.125 | [24; 392] | 0 | 370 | 630 |
| change-seasonality-weak | 64 | 1.42 @ 16 | not_confirmed | — | — | [32; 424] | 0 | 339 | 661 |
| change-seasonality-medium | 64 | 1.51 @ 0 | recovered | 232 | 3.625 | [40; 392] | 0 | 217 | 783 |
| change-seasonality-strong | 64 | 1.45 @ 0 | recovered | 48 | 0.750 | [24; 256] | 0 | 892 | 108 |
| change-frequency-weak | 64 | 1.13 @ 152 | not_confirmed | — | — | [168; 440] | 0 | 135 | 865 |
| change-frequency-medium | 64 | 1.21 @ 456 | not_confirmed | — | — | [344; 344] | 0 | 1 | 999 |
| change-frequency-strong | 64 | 1.19 @ 16 | recovered | 232 | 3.625 | [32; 440] | 8 | 624 | 368 |
| change-sensor_bias-weak | 64 | 1.09 @ 72 | recovered | 88 | 1.375 | [88; 368] | 44 | 720 | 236 |
| change-sensor_bias-medium | 64 | 1.27 @ 464 | not_confirmed | — | — | — | 0 | 0 | 1000 |
| change-sensor_bias-strong | 64 | 1.46 @ 40 | not_confirmed | — | — | [389.6; 432] | 0 | 15 | 985 |
| drift-mean-weak ≈ | 64 | 1.27 @ 368 | not_confirmed | — | — | [384; 433] | 0 | 156 | 844 |
| drift-mean-medium ≈ | 64 | 1.20 @ 64 | not_confirmed | — | — | [82.4; 437.6] | 0 | 293 | 707 |
| drift-mean-strong ≈ | 64 | 1.17 @ 32 | recovered | 368 | 5.750 | [88; 440] | 8 | 876 | 116 |
| drift-frequency-weak ≈ | 64 | 1.16 @ 152 | not_confirmed | — | — | [160; 438] | 0 | 251 | 749 |
| drift-frequency-medium ≈ | 64 | 1.10 @ 456 | not_confirmed | — | — | [160; 432] | 47 | 203 | 750 |
| drift-frequency-strong ≈ | 64 | 1.04 @ 504 | no_degradation | — | — | [64; 432] | 219 | 379 | 402 |
| anomaly-spike-weak | 64 | 1.45 @ 0 | recovered | 424 | 6.625 | [16; 440] | 0 | 335 | 665 |
| anomaly-spike-medium | 64 | 1.79 @ 0 | not_confirmed | — | — | [16; 440] | 0 | 177 | 823 |
| anomaly-spike-strong | 64 | 3.14 @ 0 | recovered | 440 | 6.875 | [208; 440] | 0 | 296 | 704 |
| anomaly-freeze-weak | 64 | 1.22 @ 56 | not_confirmed | — | — | [378.6; 429.4] | 0 | 14 | 986 |
| anomaly-freeze-medium | 64 | 1.67 @ 56 | not_confirmed | — | — | [232; 440] | 0 | 137 | 863 |
| anomaly-freeze-strong | 64 | 1.96 @ 136 | not_confirmed | — | — | [168; 433] | 0 | 236 | 764 |
| none | 64 | 1.22 @ 192 | recovered | 352 | 5.500 | [112; 440] | 4 | 429 | 567 |

## H1: H=16, Q90 (чувствительность)

CI в таблице — только по recovered-повторам. Drift помечен `≈`.

| Ячейка | L | Сглаженный пик @ d | Статус | d* | d*/L | Условный CI95 d* | boot: нет дегр. | boot: recovered | boot: не подтв. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| change-mean-weak | 512 | 1.18 @ 216 | not_confirmed | — | — | [144; 440] | 0 | 189 | 811 |
| change-mean-medium | 512 | 1.31 @ 0 | recovered | 440 | 0.859 | [304; 440] | 0 | 596 | 404 |
| change-mean-strong | 512 | 1.70 @ 0 | not_confirmed | — | — | [56; 434] | 0 | 151 | 849 |
| change-variance-weak | 512 | 1.21 @ 280 | not_confirmed | — | — | [48; 384] | 1 | 73 | 926 |
| change-variance-medium | 512 | 1.19 @ 264 | not_confirmed | — | — | [272; 440] | 0 | 352 | 648 |
| change-variance-strong | 512 | 1.17 @ 504 | not_confirmed | — | — | [92.8; 328] | 0 | 425 | 575 |
| change-trend-weak | 512 | 1.26 @ 448 | not_confirmed | — | — | — | 0 | 0 | 1000 |
| change-trend-medium | 512 | 1.25 @ 224 | recovered | 328 | 0.641 | [112; 352] | 0 | 529 | 471 |
| change-trend-strong | 512 | 1.38 @ 72 | recovered | 232 | 0.453 | [152; 424] | 0 | 515 | 485 |
| change-ar-weak | 512 | 1.18 @ 32 | recovered | 216 | 0.422 | [208; 368] | 0 | 490 | 510 |
| change-ar-medium | 512 | 1.14 @ 312 | recovered | 328 | 0.641 | [115; 440] | 1 | 456 | 543 |
| change-ar-strong | 512 | 1.11 @ 24 | recovered | 32 | 0.062 | [32; 352] | 6 | 615 | 379 |
| change-seasonality-weak | 512 | 1.37 @ 8 | not_confirmed | — | — | [32; 408] | 0 | 389 | 611 |
| change-seasonality-medium | 512 | 1.33 @ 0 | recovered | 312 | 0.609 | [48; 393.6] | 0 | 433 | 567 |
| change-seasonality-strong | 512 | 1.63 @ 0 | recovered | 200 | 0.391 | [32; 408] | 0 | 910 | 90 |
| change-frequency-weak | 512 | 1.46 @ 144 | not_confirmed | — | — | [288; 433.8] | 0 | 152 | 848 |
| change-frequency-medium | 512 | 1.78 @ 56 | not_confirmed | — | — | [299.2; 420.8] | 0 | 2 | 998 |
| change-frequency-strong | 512 | 2.19 @ 56 | recovered | 368 | 0.719 | [264; 408] | 0 | 674 | 326 |
| change-sensor_bias-weak | 512 | 1.12 @ 168 | recovered | 184 | 0.359 | [64; 400] | 1 | 619 | 380 |
| change-sensor_bias-medium | 512 | 1.18 @ 0 | not_confirmed | — | — | [80; 432] | 0 | 101 | 899 |
| change-sensor_bias-strong | 512 | 2.00 @ 0 | not_confirmed | — | — | [376; 424] | 0 | 80 | 920 |
| drift-mean-weak ≈ | 512 | 1.22 @ 368 | recovered | 432 | 0.844 | [376; 440] | 0 | 371 | 629 |
| drift-mean-medium ≈ | 512 | 1.34 @ 48 | recovered | 280 | 0.547 | [88; 440] | 0 | 524 | 476 |
| drift-mean-strong ≈ | 512 | 1.29 @ 88 | recovered | 360 | 0.703 | [182; 432] | 0 | 711 | 289 |
| drift-frequency-weak ≈ | 512 | 1.40 @ 264 | recovered | 352 | 0.688 | [321.2; 440] | 0 | 684 | 316 |
| drift-frequency-medium ≈ | 512 | 1.49 @ 128 | not_confirmed | — | — | [328; 368] | 0 | 13 | 987 |
| drift-frequency-strong ≈ | 512 | 1.78 @ 176 | not_confirmed | — | — | [376; 430.4] | 0 | 289 | 711 |
| anomaly-spike-weak | 512 | 1.25 @ 184 | not_confirmed | — | — | [200; 440] | 0 | 379 | 621 |
| anomaly-spike-medium | 512 | 1.38 @ 0 | recovered | 120 | 0.234 | [16; 440] | 1 | 670 | 329 |
| anomaly-spike-strong | 512 | 2.11 @ 0 | recovered | 416 | 0.812 | [200; 440] | 0 | 386 | 614 |
| anomaly-freeze-weak | 512 | 1.13 @ 0 | recovered | 8 | 0.016 | [8; 376] | 35 | 388 | 577 |
| anomaly-freeze-medium | 512 | 1.97 @ 0 | not_confirmed | — | — | [304; 440] | 0 | 41 | 959 |
| anomaly-freeze-strong | 512 | 1.57 @ 0 | not_confirmed | — | — | [16; 424] | 0 | 94 | 906 |
| none | 512 | 1.15 @ 80 | recovered | 136 | 0.266 | [80; 416] | 56 | 424 | 520 |
| change-mean-weak | 256 | 1.29 @ 216 | not_confirmed | — | — | [256; 440] | 0 | 57 | 943 |
| change-mean-medium | 256 | 1.23 @ 280 | recovered | 304 | 1.188 | [16; 440] | 0 | 754 | 246 |
| change-mean-strong | 256 | 1.63 @ 0 | not_confirmed | — | — | [64; 416] | 0 | 231 | 769 |
| change-variance-weak | 256 | 1.20 @ 280 | not_confirmed | — | — | [296; 376] | 0 | 115 | 885 |
| change-variance-medium | 256 | 1.21 @ 416 | not_confirmed | — | — | [280; 440] | 0 | 76 | 924 |
| change-variance-strong | 256 | 1.16 @ 504 | not_confirmed | — | — | [104; 408] | 0 | 465 | 535 |
| change-trend-weak | 256 | 1.29 @ 448 | not_confirmed | — | — | — | 0 | 0 | 1000 |
| change-trend-medium | 256 | 1.16 @ 216 | recovered | 360 | 1.406 | [99.2; 376] | 0 | 537 | 463 |
| change-trend-strong | 256 | 1.25 @ 72 | not_confirmed | — | — | [96.4; 424] | 0 | 403 | 597 |
| change-ar-weak | 256 | 1.16 @ 344 | not_confirmed | — | — | [216; 376] | 0 | 141 | 859 |
| change-ar-medium | 256 | 1.13 @ 0 | not_confirmed | — | — | [80; 419.6] | 4 | 258 | 738 |
| change-ar-strong | 256 | 1.13 @ 24 | recovered | 152 | 0.594 | [32; 368] | 0 | 620 | 380 |
| change-seasonality-weak | 256 | 1.44 @ 16 | recovered | 112 | 0.438 | [40; 424] | 0 | 575 | 425 |
| change-seasonality-medium | 256 | 1.44 @ 0 | recovered | 96 | 0.375 | [48; 432] | 0 | 479 | 521 |
| change-seasonality-strong | 256 | 1.57 @ 0 | recovered | 88 | 0.344 | [32; 272] | 0 | 935 | 65 |
| change-frequency-weak | 256 | 1.23 @ 144 | not_confirmed | — | — | [178.8; 440] | 0 | 55 | 945 |
| change-frequency-medium | 256 | 1.54 @ 56 | not_confirmed | — | — | [72; 328] | 0 | 52 | 948 |
| change-frequency-strong | 256 | 1.99 @ 56 | not_confirmed | — | — | [168; 392] | 0 | 302 | 698 |
| change-sensor_bias-weak | 256 | 1.13 @ 168 | recovered | 184 | 0.719 | [56; 437.6] | 22 | 733 | 245 |
| change-sensor_bias-medium | 256 | 1.20 @ 0 | not_confirmed | — | — | [12.6; 344.6] | 0 | 24 | 976 |
| change-sensor_bias-strong | 256 | 1.79 @ 0 | not_confirmed | — | — | [224; 416] | 0 | 136 | 864 |
| drift-mean-weak ≈ | 256 | 1.24 @ 328 | recovered | 400 | 1.562 | [376; 440] | 0 | 391 | 609 |
| drift-mean-medium ≈ | 256 | 1.34 @ 48 | not_confirmed | — | — | [245; 432] | 0 | 254 | 746 |
| drift-mean-strong ≈ | 256 | 1.32 @ 88 | recovered | 400 | 1.562 | [272; 432] | 0 | 457 | 543 |
| drift-frequency-weak ≈ | 256 | 1.14 @ 144 | not_confirmed | — | — | [154.6; 336] | 1 | 334 | 665 |
| drift-frequency-medium ≈ | 256 | 1.29 @ 120 | not_confirmed | — | — | [136; 184] | 0 | 218 | 782 |
| drift-frequency-strong ≈ | 256 | 1.57 @ 72 | not_confirmed | — | — | [184; 440] | 0 | 131 | 869 |
| anomaly-spike-weak | 256 | 1.17 @ 264 | recovered | 280 | 1.094 | [40; 424] | 0 | 499 | 501 |
| anomaly-spike-medium | 256 | 1.50 @ 0 | not_confirmed | — | — | [20.6; 440] | 0 | 344 | 656 |
| anomaly-spike-strong | 256 | 2.18 @ 0 | not_confirmed | — | — | [408; 440] | 0 | 206 | 794 |
| anomaly-freeze-weak | 256 | 1.08 @ 0 | recovered | 8 | 0.031 | [8; 392] | 14 | 482 | 504 |
| anomaly-freeze-medium | 256 | 1.64 @ 0 | not_confirmed | — | — | [224; 440] | 0 | 15 | 985 |
| anomaly-freeze-strong | 256 | 1.63 @ 0 | not_confirmed | — | — | [344; 416] | 0 | 49 | 951 |
| none | 256 | 1.20 @ 80 | recovered | 392 | 1.531 | [90.8; 429.2] | 6 | 615 | 379 |
| change-mean-weak | 128 | 1.29 @ 216 | not_confirmed | — | — | [240; 436] | 0 | 181 | 819 |
| change-mean-medium | 128 | 1.22 @ 288 | recovered | 304 | 2.375 | [48; 440] | 0 | 758 | 242 |
| change-mean-strong | 128 | 1.33 @ 0 | recovered | 144 | 1.125 | [16; 440] | 0 | 602 | 398 |
| change-variance-weak | 128 | 1.20 @ 280 | recovered | 304 | 2.375 | [56; 376] | 19 | 547 | 434 |
| change-variance-medium | 128 | 1.19 @ 264 | recovered | 288 | 2.250 | [176; 432] | 0 | 781 | 219 |
| change-variance-strong | 128 | 1.17 @ 208 | recovered | 224 | 1.750 | [80; 416.4] | 49 | 679 | 272 |
| change-trend-weak | 128 | 1.27 @ 216 | not_confirmed | — | — | — | 0 | 0 | 1000 |
| change-trend-medium | 128 | 1.32 @ 224 | recovered | 360 | 2.812 | [240; 424] | 0 | 515 | 485 |
| change-trend-strong | 128 | 1.30 @ 368 | not_confirmed | — | — | [328; 432] | 0 | 205 | 795 |
| change-ar-weak | 128 | 1.22 @ 0 | not_confirmed | — | — | [61.2; 368] | 0 | 227 | 773 |
| change-ar-medium | 128 | 1.26 @ 8 | not_confirmed | — | — | [16; 440] | 1 | 300 | 699 |
| change-ar-strong | 128 | 1.16 @ 16 | recovered | 32 | 0.250 | [32; 280] | 9 | 847 | 144 |
| change-seasonality-weak | 128 | 1.51 @ 16 | recovered | 112 | 0.875 | [32; 424] | 0 | 747 | 253 |
| change-seasonality-medium | 128 | 1.45 @ 0 | recovered | 96 | 0.750 | [24; 376] | 0 | 499 | 501 |
| change-seasonality-strong | 128 | 1.53 @ 0 | recovered | 88 | 0.688 | [32; 264] | 0 | 844 | 156 |
| change-frequency-weak | 128 | 1.14 @ 168 | recovered | 184 | 1.438 | [152; 432] | 8 | 279 | 713 |
| change-frequency-medium | 128 | 1.27 @ 56 | not_confirmed | — | — | [64; 281.4] | 0 | 194 | 806 |
| change-frequency-strong | 128 | 1.44 @ 16 | recovered | 144 | 1.125 | [72; 384] | 0 | 839 | 161 |
| change-sensor_bias-weak | 128 | 1.23 @ 176 | recovered | 184 | 1.438 | [88; 408] | 2 | 879 | 119 |
| change-sensor_bias-medium | 128 | 1.20 @ 88 | recovered | 96 | 0.750 | [8; 416] | 5 | 327 | 668 |
| change-sensor_bias-strong | 128 | 1.63 @ 0 | not_confirmed | — | — | [175.4; 440] | 0 | 80 | 920 |
| drift-mean-weak ≈ | 128 | 1.27 @ 368 | recovered | 392 | 3.062 | [376; 440] | 0 | 642 | 358 |
| drift-mean-medium ≈ | 128 | 1.19 @ 80 | recovered | 168 | 1.312 | [72; 440] | 0 | 597 | 403 |
| drift-mean-strong ≈ | 128 | 1.19 @ 344 | recovered | 368 | 2.875 | [88; 440] | 0 | 795 | 205 |
| drift-frequency-weak ≈ | 128 | 1.17 @ 152 | recovered | 168 | 1.312 | [160; 440] | 1 | 479 | 520 |
| drift-frequency-medium ≈ | 128 | 1.20 @ 456 | not_confirmed | — | — | [136; 376] | 5 | 62 | 933 |
| drift-frequency-strong ≈ | 128 | 1.16 @ 48 | recovered | 88 | 0.688 | [56; 440] | 72 | 650 | 278 |
| anomaly-spike-weak | 128 | 1.42 @ 0 | recovered | 32 | 0.250 | [16; 440] | 0 | 547 | 453 |
| anomaly-spike-medium | 128 | 1.71 @ 0 | not_confirmed | — | — | [16; 414.4] | 0 | 343 | 657 |
| anomaly-spike-strong | 128 | 3.03 @ 0 | not_confirmed | — | — | [208; 440] | 0 | 278 | 722 |
| anomaly-freeze-weak | 128 | 1.18 @ 264 | recovered | 272 | 2.125 | [8; 392] | 9 | 476 | 515 |
| anomaly-freeze-medium | 128 | 1.22 @ 56 | not_confirmed | — | — | [224; 440] | 0 | 307 | 693 |
| anomaly-freeze-strong | 128 | 2.07 @ 136 | not_confirmed | — | — | [176; 440] | 0 | 225 | 775 |
| none | 128 | 1.08 @ 120 | no_degradation | — | — | [128; 424] | 168 | 658 | 174 |
| change-mean-weak | 64 | 1.27 @ 232 | not_confirmed | — | — | [288; 432] | 0 | 61 | 939 |
| change-mean-medium | 64 | 1.21 @ 288 | not_confirmed | — | — | [304; 440] | 0 | 154 | 846 |
| change-mean-strong | 64 | 1.35 @ 0 | not_confirmed | — | — | [88; 432] | 0 | 196 | 804 |
| change-variance-weak | 64 | 1.25 @ 40 | not_confirmed | — | — | [48; 440] | 3 | 74 | 923 |
| change-variance-medium | 64 | 1.18 @ 48 | recovered | 432 | 6.750 | [280; 432.8] | 0 | 437 | 563 |
| change-variance-strong | 64 | 1.17 @ 216 | recovered | 240 | 3.750 | [112; 432] | 5 | 574 | 421 |
| change-trend-weak | 64 | 1.33 @ 384 | not_confirmed | — | — | [400; 400] | 0 | 3 | 997 |
| change-trend-medium | 64 | 1.20 @ 48 | not_confirmed | — | — | [376; 429.2] | 0 | 55 | 945 |
| change-trend-strong | 64 | 1.10 @ 224 | recovered | 408 | 6.375 | [104; 424] | 0 | 656 | 344 |
| change-ar-weak | 64 | 1.24 @ 504 | not_confirmed | — | — | — | 0 | 0 | 1000 |
| change-ar-medium | 64 | 1.26 @ 0 | not_confirmed | — | — | [278.6; 440] | 0 | 34 | 966 |
| change-ar-strong | 64 | 1.12 @ 16 | not_confirmed | — | — | [36.8; 416] | 0 | 225 | 775 |
| change-seasonality-weak | 64 | 1.42 @ 16 | not_confirmed | — | — | [32; 424] | 0 | 259 | 741 |
| change-seasonality-medium | 64 | 1.51 @ 0 | not_confirmed | — | — | [84; 392] | 0 | 109 | 891 |
| change-seasonality-strong | 64 | 1.45 @ 0 | recovered | 56 | 0.875 | [24; 256] | 0 | 812 | 188 |
| change-frequency-weak | 64 | 1.13 @ 152 | not_confirmed | — | — | [168; 440] | 0 | 63 | 937 |
| change-frequency-medium | 64 | 1.21 @ 456 | not_confirmed | — | — | — | 0 | 0 | 1000 |
| change-frequency-strong | 64 | 1.19 @ 16 | recovered | 280 | 4.375 | [32; 440] | 3 | 461 | 536 |
| change-sensor_bias-weak | 64 | 1.09 @ 72 | recovered | 200 | 3.125 | [88; 376] | 18 | 685 | 297 |
| change-sensor_bias-medium | 64 | 1.27 @ 464 | not_confirmed | — | — | — | 0 | 0 | 1000 |
| change-sensor_bias-strong | 64 | 1.46 @ 40 | not_confirmed | — | — | [400.4; 415.6] | 0 | 3 | 997 |
| drift-mean-weak ≈ | 64 | 1.27 @ 368 | not_confirmed | — | — | [384.6; 439.4] | 0 | 84 | 916 |
| drift-mean-medium ≈ | 64 | 1.20 @ 64 | not_confirmed | — | — | [88; 437.2] | 0 | 215 | 785 |
| drift-mean-strong ≈ | 64 | 1.17 @ 32 | recovered | 376 | 5.875 | [112; 440] | 1 | 802 | 197 |
| drift-frequency-weak ≈ | 64 | 1.16 @ 152 | not_confirmed | — | — | [176; 440] | 0 | 190 | 810 |
| drift-frequency-medium ≈ | 64 | 1.10 @ 456 | not_confirmed | — | — | [160; 432] | 20 | 185 | 795 |
| drift-frequency-strong ≈ | 64 | 1.04 @ 504 | not_confirmed | — | — | [64; 432] | 120 | 392 | 488 |
| anomaly-spike-weak | 64 | 1.45 @ 0 | not_confirmed | — | — | [16; 440] | 0 | 254 | 746 |
| anomaly-spike-medium | 64 | 1.79 @ 0 | not_confirmed | — | — | [48; 440] | 0 | 118 | 882 |
| anomaly-spike-strong | 64 | 3.14 @ 0 | not_confirmed | — | — | [344; 440] | 0 | 204 | 796 |
| anomaly-freeze-weak | 64 | 1.22 @ 56 | not_confirmed | — | — | [385.6; 431.2] | 0 | 3 | 997 |
| anomaly-freeze-medium | 64 | 1.67 @ 56 | not_confirmed | — | — | [312; 440] | 0 | 80 | 920 |
| anomaly-freeze-strong | 64 | 1.96 @ 136 | not_confirmed | — | — | [168; 424.6] | 0 | 158 | 842 |
| none | 64 | 1.22 @ 192 | recovered | 352 | 5.500 | [144; 440] | 2 | 382 | 616 |

## H1: H=64, Q95 (основной)

CI в таблице — только по recovered-повторам. Drift помечен `≈`.

| Ячейка | L | Сглаженный пик @ d | Статус | d* | d*/L | Условный CI95 d* | boot: нет дегр. | boot: recovered | boot: не подтв. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| change-mean-weak | 512 | 1.12 @ 504 | not_confirmed | — | — | [40; 424] | 28 | 155 | 817 |
| change-mean-medium | 512 | 1.27 @ 0 | recovered | 440 | 0.859 | [128; 440] | 0 | 567 | 433 |
| change-mean-strong | 512 | 1.90 @ 0 | recovered | 400 | 0.781 | [360; 440] | 0 | 526 | 474 |
| change-variance-weak | 512 | 1.08 @ 48 | recovered | 256 | 0.500 | [56; 440] | 155 | 449 | 396 |
| change-variance-medium | 512 | 1.07 @ 240 | recovered | 264 | 0.516 | [48; 440] | 98 | 863 | 39 |
| change-variance-strong | 512 | 1.04 @ 64 | no_degradation | — | — | [72; 312] | 61 | 658 | 281 |
| change-trend-weak | 512 | 1.18 @ 440 | not_confirmed | — | — | [248; 256] | 0 | 5 | 995 |
| change-trend-medium | 512 | 1.59 @ 120 | recovered | 328 | 0.641 | [304; 344] | 0 | 601 | 399 |
| change-trend-strong | 512 | 2.22 @ 80 | recovered | 416 | 0.812 | [308.6; 427.4] | 0 | 424 | 576 |
| change-ar-weak | 512 | 1.12 @ 16 | recovered | 216 | 0.422 | [25; 264] | 12 | 606 | 382 |
| change-ar-medium | 512 | 1.09 @ 264 | recovered | 320 | 0.625 | [120; 440] | 0 | 876 | 124 |
| change-ar-strong | 512 | 1.05 @ 24 | recovered | 32 | 0.062 | [24; 432] | 82 | 419 | 499 |
| change-seasonality-weak | 512 | 1.26 @ 8 | recovered | 280 | 0.547 | [32; 440] | 0 | 452 | 548 |
| change-seasonality-medium | 512 | 1.35 @ 0 | recovered | 240 | 0.469 | [56; 352] | 0 | 689 | 311 |
| change-seasonality-strong | 512 | 1.48 @ 0 | recovered | 168 | 0.328 | [48; 224] | 0 | 995 | 5 |
| change-frequency-weak | 512 | 1.42 @ 144 | not_confirmed | — | — | [288; 440] | 0 | 106 | 894 |
| change-frequency-medium | 512 | 1.60 @ 48 | not_confirmed | — | — | [280; 428] | 0 | 4 | 996 |
| change-frequency-strong | 512 | 1.72 @ 16 | recovered | 312 | 0.609 | [280; 440] | 0 | 521 | 479 |
| change-sensor_bias-weak | 512 | 1.10 @ 40 | not_confirmed | — | — | [56; 344] | 2 | 318 | 680 |
| change-sensor_bias-medium | 512 | 1.29 @ 0 | not_confirmed | — | — | [48; 416] | 0 | 83 | 917 |
| change-sensor_bias-strong | 512 | 2.03 @ 0 | not_confirmed | — | — | [368; 424] | 0 | 98 | 902 |
| drift-mean-weak ≈ | 512 | 1.16 @ 320 | recovered | 376 | 0.734 | [336; 440] | 1 | 755 | 244 |
| drift-mean-medium ≈ | 512 | 1.25 @ 48 | recovered | 256 | 0.500 | [104; 440] | 0 | 930 | 70 |
| drift-mean-strong ≈ | 512 | 1.61 @ 88 | recovered | 352 | 0.688 | [288; 376.4] | 0 | 639 | 361 |
| drift-frequency-weak ≈ | 512 | 1.27 @ 224 | recovered | 336 | 0.656 | [304; 440] | 0 | 863 | 137 |
| drift-frequency-medium ≈ | 512 | 1.42 @ 112 | not_confirmed | — | — | [424.8; 439.2] | 0 | 5 | 995 |
| drift-frequency-strong ≈ | 512 | 1.57 @ 128 | not_confirmed | — | — | [344; 440] | 0 | 137 | 863 |
| anomaly-spike-weak | 512 | 1.09 @ 224 | recovered | 256 | 0.500 | [8; 424] | 118 | 562 | 320 |
| anomaly-spike-medium | 512 | 1.23 @ 0 | recovered | 424 | 0.828 | [16; 432] | 15 | 437 | 548 |
| anomaly-spike-strong | 512 | 1.59 @ 0 | recovered | 408 | 0.797 | [16; 432] | 0 | 819 | 181 |
| anomaly-freeze-weak | 512 | 1.07 @ 472 | not_confirmed | — | — | [8; 352] | 186 | 208 | 606 |
| anomaly-freeze-medium | 512 | 1.58 @ 0 | not_confirmed | — | — | [112; 440] | 0 | 412 | 588 |
| anomaly-freeze-strong | 512 | 1.73 @ 0 | not_confirmed | — | — | [152; 438.6] | 0 | 168 | 832 |
| none | 512 | 1.05 @ 72 | recovered | 80 | 0.156 | [56; 440] | 218 | 323 | 459 |
| change-mean-weak | 256 | 1.21 @ 504 | not_confirmed | — | — | [52.8; 424] | 28 | 129 | 843 |
| change-mean-medium | 256 | 1.17 @ 0 | recovered | 16 | 0.062 | [8; 304] | 98 | 869 | 33 |
| change-mean-strong | 256 | 1.69 @ 0 | recovered | 240 | 0.938 | [50; 360] | 0 | 846 | 154 |
| change-variance-weak | 256 | 1.10 @ 240 | recovered | 272 | 1.062 | [56; 432] | 265 | 663 | 72 |
| change-variance-medium | 256 | 1.11 @ 248 | recovered | 288 | 1.125 | [48; 440] | 33 | 953 | 14 |
| change-variance-strong | 256 | 1.04 @ 200 | no_degradation | — | — | [64; 400] | 416 | 441 | 143 |
| change-trend-weak | 256 | 1.23 @ 448 | not_confirmed | — | — | [224; 256] | 0 | 31 | 969 |
| change-trend-medium | 256 | 1.18 @ 296 | recovered | 312 | 1.219 | [240.2; 328] | 47 | 642 | 311 |
| change-trend-strong | 256 | 1.39 @ 72 | not_confirmed | — | — | [320; 432] | 0 | 173 | 827 |
| change-ar-weak | 256 | 1.08 @ 8 | no_degradation | — | — | [8; 360] | 372 | 425 | 203 |
| change-ar-medium | 256 | 1.09 @ 72 | recovered | 80 | 0.312 | [64.8; 440] | 296 | 683 | 21 |
| change-ar-strong | 256 | 1.07 @ 16 | no_degradation | — | — | [8; 361] | 596 | 376 | 28 |
| change-seasonality-weak | 256 | 1.30 @ 16 | recovered | 32 | 0.125 | [24; 440] | 0 | 747 | 253 |
| change-seasonality-medium | 256 | 1.41 @ 0 | recovered | 64 | 0.250 | [48; 320] | 0 | 899 | 101 |
| change-seasonality-strong | 256 | 1.51 @ 0 | recovered | 56 | 0.219 | [32; 200] | 0 | 1000 | 0 |
| change-frequency-weak | 256 | 1.27 @ 112 | not_confirmed | — | — | [152; 432] | 0 | 347 | 653 |
| change-frequency-medium | 256 | 1.27 @ 48 | not_confirmed | — | — | [64; 264] | 1 | 446 | 553 |
| change-frequency-strong | 256 | 1.52 @ 16 | recovered | 192 | 0.750 | [140.8; 264] | 0 | 945 | 55 |
| change-sensor_bias-weak | 256 | 1.12 @ 40 | recovered | 56 | 0.219 | [48; 344] | 32 | 558 | 410 |
| change-sensor_bias-medium | 256 | 1.24 @ 0 | not_confirmed | — | — | [16; 440] | 0 | 408 | 592 |
| change-sensor_bias-strong | 256 | 1.71 @ 0 | not_confirmed | — | — | [104; 408] | 0 | 397 | 603 |
| drift-mean-weak ≈ | 256 | 1.15 @ 328 | recovered | 352 | 1.375 | [40; 424] | 13 | 916 | 71 |
| drift-mean-medium ≈ | 256 | 1.24 @ 48 | recovered | 272 | 1.062 | [64; 440] | 0 | 964 | 36 |
| drift-mean-strong ≈ | 256 | 1.48 @ 88 | recovered | 336 | 1.312 | [280; 392] | 0 | 957 | 43 |
| drift-frequency-weak ≈ | 256 | 1.09 @ 144 | no_degradation | — | — | [152; 320] | 144 | 305 | 551 |
| drift-frequency-medium ≈ | 256 | 1.25 @ 112 | recovered | 136 | 0.531 | [128; 168] | 0 | 466 | 534 |
| drift-frequency-strong ≈ | 256 | 1.42 @ 120 | not_confirmed | — | — | [168; 344] | 0 | 314 | 686 |
| anomaly-spike-weak | 256 | 1.07 @ 0 | no_degradation | — | — | [8; 288] | 364 | 412 | 224 |
| anomaly-spike-medium | 256 | 1.30 @ 0 | recovered | 24 | 0.094 | [9.8; 432] | 11 | 650 | 339 |
| anomaly-spike-strong | 256 | 1.61 @ 0 | recovered | 384 | 1.500 | [16; 432.6] | 0 | 838 | 162 |
| anomaly-freeze-weak | 256 | 1.08 @ 8 | no_degradation | — | — | [8; 408] | 236 | 398 | 366 |
| anomaly-freeze-medium | 256 | 1.56 @ 40 | not_confirmed | — | — | [112; 440] | 0 | 425 | 575 |
| anomaly-freeze-strong | 256 | 1.66 @ 0 | recovered | 400 | 1.562 | [150; 432] | 0 | 511 | 489 |
| none | 256 | 1.10 @ 112 | recovered | 120 | 0.469 | [69.6; 392] | 380 | 528 | 92 |
| change-mean-weak | 128 | 1.19 @ 216 | not_confirmed | — | — | [232; 432] | 0 | 179 | 821 |
| change-mean-medium | 128 | 1.22 @ 504 | not_confirmed | — | — | [40; 313.4] | 0 | 34 | 966 |
| change-mean-strong | 128 | 1.42 @ 0 | not_confirmed | — | — | [88; 440] | 0 | 253 | 747 |
| change-variance-weak | 128 | 1.15 @ 48 | not_confirmed | — | — | [64; 432] | 3 | 151 | 846 |
| change-variance-medium | 128 | 1.19 @ 280 | recovered | 424 | 3.312 | [304; 440] | 0 | 584 | 416 |
| change-variance-strong | 128 | 1.11 @ 216 | recovered | 240 | 1.875 | [72; 432] | 7 | 875 | 118 |
| change-trend-weak | 128 | 1.19 @ 208 | not_confirmed | — | — | [240; 240] | 0 | 2 | 998 |
| change-trend-medium | 128 | 1.22 @ 280 | recovered | 360 | 2.812 | [310.6; 432] | 0 | 394 | 606 |
| change-trend-strong | 128 | 1.35 @ 360 | not_confirmed | — | — | [378.4; 440] | 0 | 85 | 915 |
| change-ar-weak | 128 | 1.27 @ 0 | not_confirmed | — | — | [232; 352] | 0 | 132 | 868 |
| change-ar-medium | 128 | 1.19 @ 24 | not_confirmed | — | — | [320.8; 440] | 0 | 52 | 948 |
| change-ar-strong | 128 | 1.16 @ 0 | not_confirmed | — | — | [120; 432] | 0 | 320 | 680 |
| change-seasonality-weak | 128 | 1.34 @ 16 | not_confirmed | — | — | [32; 424] | 0 | 238 | 762 |
| change-seasonality-medium | 128 | 1.37 @ 0 | not_confirmed | — | — | [64; 400] | 0 | 344 | 656 |
| change-seasonality-strong | 128 | 1.46 @ 8 | recovered | 88 | 0.688 | [48; 432] | 0 | 676 | 324 |
| change-frequency-weak | 128 | 1.17 @ 104 | not_confirmed | — | — | [112; 416] | 0 | 67 | 933 |
| change-frequency-medium | 128 | 1.16 @ 48 | not_confirmed | — | — | [59.2; 273.6] | 0 | 57 | 943 |
| change-frequency-strong | 128 | 1.27 @ 16 | recovered | 184 | 1.438 | [72; 424] | 0 | 596 | 404 |
| change-sensor_bias-weak | 128 | 1.20 @ 176 | recovered | 344 | 2.688 | [80; 352] | 0 | 468 | 532 |
| change-sensor_bias-medium | 128 | 1.19 @ 0 | not_confirmed | — | — | [136.4; 424] | 0 | 30 | 970 |
| change-sensor_bias-strong | 128 | 1.44 @ 0 | not_confirmed | — | — | [112; 374] | 0 | 46 | 954 |
| drift-mean-weak ≈ | 128 | 1.23 @ 328 | not_confirmed | — | — | [360; 440] | 0 | 385 | 615 |
| drift-mean-medium ≈ | 128 | 1.17 @ 48 | not_confirmed | — | — | [72; 440] | 0 | 215 | 785 |
| drift-mean-strong ≈ | 128 | 1.30 @ 88 | not_confirmed | — | — | [280; 416] | 0 | 263 | 737 |
| drift-frequency-weak ≈ | 128 | 1.05 @ 432 | not_confirmed | — | — | [144; 440] | 46 | 390 | 564 |
| drift-frequency-medium ≈ | 128 | 1.21 @ 104 | not_confirmed | — | — | [163; 257] | 0 | 6 | 994 |
| drift-frequency-strong ≈ | 128 | 1.25 @ 48 | not_confirmed | — | — | [104; 440] | 0 | 21 | 979 |
| anomaly-spike-weak | 128 | 1.30 @ 0 | recovered | 32 | 0.250 | [16; 304] | 16 | 507 | 477 |
| anomaly-spike-medium | 128 | 1.52 @ 0 | recovered | 80 | 0.625 | [8; 424] | 6 | 501 | 493 |
| anomaly-spike-strong | 128 | 2.36 @ 0 | not_confirmed | — | — | [408; 408] | 0 | 2 | 998 |
| anomaly-freeze-weak | 128 | 1.13 @ 256 | not_confirmed | — | — | [24; 424] | 1 | 110 | 889 |
| anomaly-freeze-medium | 128 | 1.43 @ 48 | not_confirmed | — | — | [246; 440] | 0 | 131 | 869 |
| anomaly-freeze-strong | 128 | 2.56 @ 128 | not_confirmed | — | — | [208; 437.8] | 0 | 52 | 948 |
| none | 128 | 1.01 @ 304 | no_degradation | — | — | [32; 440] | 232 | 642 | 126 |
| change-mean-weak | 64 | 1.16 @ 504 | not_confirmed | — | — | [136; 304] | 3 | 67 | 930 |
| change-mean-medium | 64 | 1.26 @ 504 | not_confirmed | — | — | — | 0 | 0 | 1000 |
| change-mean-strong | 64 | 1.35 @ 0 | not_confirmed | — | — | [21.6; 432] | 0 | 135 | 865 |
| change-variance-weak | 64 | 1.12 @ 48 | not_confirmed | — | — | [56; 440] | 27 | 339 | 634 |
| change-variance-medium | 64 | 1.18 @ 272 | recovered | 304 | 4.750 | [288; 432] | 0 | 785 | 215 |
| change-variance-strong | 64 | 1.12 @ 216 | recovered | 240 | 3.750 | [80; 440] | 2 | 760 | 238 |
| change-trend-weak | 64 | 1.16 @ 448 | not_confirmed | — | — | [232; 232] | 0 | 3 | 997 |
| change-trend-medium | 64 | 1.11 @ 272 | recovered | 328 | 5.125 | [192; 408] | 0 | 847 | 153 |
| change-trend-strong | 64 | 1.11 @ 224 | recovered | 384 | 6.000 | [112; 440] | 0 | 593 | 407 |
| change-ar-weak | 64 | 1.31 @ 0 | not_confirmed | — | — | [280; 280] | 0 | 1 | 999 |
| change-ar-medium | 64 | 1.17 @ 272 | not_confirmed | — | — | [288; 440] | 0 | 29 | 971 |
| change-ar-strong | 64 | 1.10 @ 0 | not_confirmed | — | — | [248; 440] | 0 | 158 | 842 |
| change-seasonality-weak | 64 | 1.33 @ 16 | not_confirmed | — | — | [32; 408] | 0 | 335 | 665 |
| change-seasonality-medium | 64 | 1.45 @ 0 | not_confirmed | — | — | [64; 408] | 0 | 143 | 857 |
| change-seasonality-strong | 64 | 1.30 @ 8 | recovered | 56 | 0.875 | [32; 247.8] | 0 | 642 | 358 |
| change-frequency-weak | 64 | 1.15 @ 144 | not_confirmed | — | — | [152; 440] | 0 | 83 | 917 |
| change-frequency-medium | 64 | 1.12 @ 448 | not_confirmed | — | — | [168; 336] | 8 | 64 | 928 |
| change-frequency-strong | 64 | 1.19 @ 8 | recovered | 440 | 6.875 | [32; 440] | 0 | 440 | 560 |
| change-sensor_bias-weak | 64 | 1.06 @ 504 | not_confirmed | — | — | [80; 352] | 0 | 661 | 339 |
| change-sensor_bias-medium | 64 | 1.27 @ 456 | not_confirmed | — | — | [376; 383.4] | 0 | 4 | 996 |
| change-sensor_bias-strong | 64 | 1.34 @ 0 | not_confirmed | — | — | [216; 400] | 0 | 22 | 978 |
| drift-mean-weak ≈ | 64 | 1.22 @ 328 | not_confirmed | — | — | [352; 440] | 0 | 316 | 684 |
| drift-mean-medium ≈ | 64 | 1.12 @ 48 | recovered | 440 | 6.875 | [64; 440] | 3 | 445 | 552 |
| drift-mean-strong ≈ | 64 | 1.27 @ 32 | not_confirmed | — | — | [215; 416] | 0 | 436 | 564 |
| drift-frequency-weak ≈ | 64 | 1.09 @ 128 | recovered | 312 | 4.875 | [212.8; 440] | 0 | 465 | 535 |
| drift-frequency-medium ≈ | 64 | 1.03 @ 408 | not_confirmed | — | — | [160; 440] | 64 | 245 | 691 |
| drift-frequency-strong ≈ | 64 | 1.16 @ 40 | not_confirmed | — | — | [56; 416] | 0 | 245 | 755 |
| anomaly-spike-weak | 64 | 1.29 @ 0 | recovered | 24 | 0.375 | [8; 432] | 2 | 434 | 564 |
| anomaly-spike-medium | 64 | 1.69 @ 0 | not_confirmed | — | — | [16; 432] | 0 | 329 | 671 |
| anomaly-spike-strong | 64 | 2.49 @ 0 | not_confirmed | — | — | [361.6; 424] | 0 | 89 | 911 |
| anomaly-freeze-weak | 64 | 1.22 @ 72 | not_confirmed | — | — | [104.8; 424] | 0 | 53 | 947 |
| anomaly-freeze-medium | 64 | 1.79 @ 72 | not_confirmed | — | — | [420.4; 440] | 0 | 63 | 937 |
| anomaly-freeze-strong | 64 | 2.35 @ 128 | not_confirmed | — | — | [176; 440] | 0 | 90 | 910 |
| none | 64 | 1.11 @ 264 | not_confirmed | — | — | [112.2; 416] | 9 | 402 | 589 |

## H1: H=64, Q90 (чувствительность)

CI в таблице — только по recovered-повторам. Drift помечен `≈`.

| Ячейка | L | Сглаженный пик @ d | Статус | d* | d*/L | Условный CI95 d* | boot: нет дегр. | boot: recovered | boot: не подтв. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| change-mean-weak | 512 | 1.12 @ 504 | not_confirmed | — | — | [40; 424] | 6 | 102 | 892 |
| change-mean-medium | 512 | 1.27 @ 0 | recovered | 440 | 0.859 | [269.8; 440] | 0 | 350 | 650 |
| change-mean-strong | 512 | 1.90 @ 0 | not_confirmed | — | — | [360; 440] | 0 | 360 | 640 |
| change-variance-weak | 512 | 1.08 @ 48 | not_confirmed | — | — | [64; 440] | 77 | 349 | 574 |
| change-variance-medium | 512 | 1.07 @ 240 | recovered | 272 | 0.531 | [56; 440] | 27 | 883 | 90 |
| change-variance-strong | 512 | 1.04 @ 64 | recovered | 112 | 0.219 | [72; 320] | 1 | 548 | 451 |
| change-trend-weak | 512 | 1.18 @ 440 | not_confirmed | — | — | — | 0 | 0 | 1000 |
| change-trend-medium | 512 | 1.59 @ 120 | recovered | 328 | 0.641 | [312; 344] | 0 | 464 | 536 |
| change-trend-strong | 512 | 2.22 @ 80 | not_confirmed | — | — | [320; 434.4] | 0 | 309 | 691 |
| change-ar-weak | 512 | 1.12 @ 16 | not_confirmed | — | — | [184; 344] | 1 | 413 | 586 |
| change-ar-medium | 512 | 1.09 @ 264 | recovered | 432 | 0.844 | [320; 440] | 0 | 672 | 328 |
| change-ar-strong | 512 | 1.05 @ 24 | not_confirmed | — | — | [40.2; 432] | 1 | 202 | 797 |
| change-seasonality-weak | 512 | 1.26 @ 8 | not_confirmed | — | — | [32; 440] | 0 | 316 | 684 |
| change-seasonality-medium | 512 | 1.35 @ 0 | recovered | 240 | 0.469 | [64; 372.8] | 0 | 497 | 503 |
| change-seasonality-strong | 512 | 1.48 @ 0 | recovered | 168 | 0.328 | [64; 232] | 0 | 978 | 22 |
| change-frequency-weak | 512 | 1.42 @ 144 | not_confirmed | — | — | [296; 440] | 0 | 36 | 964 |
| change-frequency-medium | 512 | 1.60 @ 48 | not_confirmed | — | — | — | 0 | 0 | 1000 |
| change-frequency-strong | 512 | 1.72 @ 16 | not_confirmed | — | — | [296; 440] | 0 | 349 | 651 |
| change-sensor_bias-weak | 512 | 1.10 @ 40 | not_confirmed | — | — | [88; 352] | 1 | 209 | 790 |
| change-sensor_bias-medium | 512 | 1.29 @ 0 | not_confirmed | — | — | [48; 417.8] | 0 | 38 | 962 |
| change-sensor_bias-strong | 512 | 2.03 @ 0 | not_confirmed | — | — | [382.8; 425.2] | 0 | 35 | 965 |
| drift-mean-weak ≈ | 512 | 1.16 @ 320 | recovered | 432 | 0.844 | [344; 440] | 0 | 605 | 395 |
| drift-mean-medium ≈ | 512 | 1.25 @ 48 | recovered | 256 | 0.500 | [160; 440] | 0 | 843 | 157 |
| drift-mean-strong ≈ | 512 | 1.61 @ 88 | not_confirmed | — | — | [336; 376] | 0 | 443 | 557 |
| drift-frequency-weak ≈ | 512 | 1.27 @ 224 | recovered | 336 | 0.656 | [312; 440] | 0 | 722 | 278 |
| drift-frequency-medium ≈ | 512 | 1.42 @ 112 | not_confirmed | — | — | [440; 440] | 0 | 1 | 999 |
| drift-frequency-strong ≈ | 512 | 1.57 @ 128 | not_confirmed | — | — | [352; 440] | 0 | 56 | 944 |
| anomaly-spike-weak | 512 | 1.09 @ 224 | recovered | 264 | 0.516 | [16; 424] | 61 | 489 | 450 |
| anomaly-spike-medium | 512 | 1.23 @ 0 | not_confirmed | — | — | [8.4; 432] | 4 | 323 | 673 |
| anomaly-spike-strong | 512 | 1.59 @ 0 | recovered | 408 | 0.797 | [32; 436] | 0 | 701 | 299 |
| anomaly-freeze-weak | 512 | 1.07 @ 472 | not_confirmed | — | — | [8; 360] | 90 | 201 | 709 |
| anomaly-freeze-medium | 512 | 1.58 @ 0 | not_confirmed | — | — | [200; 440] | 0 | 274 | 726 |
| anomaly-freeze-strong | 512 | 1.73 @ 0 | not_confirmed | — | — | [386; 440] | 0 | 86 | 914 |
| none | 512 | 1.05 @ 72 | recovered | 128 | 0.250 | [56; 440] | 118 | 307 | 575 |
| change-mean-weak | 256 | 1.21 @ 504 | not_confirmed | — | — | [136; 424] | 3 | 74 | 923 |
| change-mean-medium | 256 | 1.17 @ 0 | recovered | 72 | 0.281 | [8; 408] | 14 | 856 | 130 |
| change-mean-strong | 256 | 1.69 @ 0 | recovered | 336 | 1.312 | [232; 400] | 0 | 644 | 356 |
| change-variance-weak | 256 | 1.10 @ 240 | recovered | 280 | 1.094 | [64; 440] | 60 | 643 | 297 |
| change-variance-medium | 256 | 1.11 @ 248 | recovered | 424 | 1.656 | [288; 440] | 0 | 913 | 87 |
| change-variance-strong | 256 | 1.04 @ 200 | no_degradation | — | — | [72; 400] | 120 | 611 | 269 |
| change-trend-weak | 256 | 1.23 @ 448 | not_confirmed | — | — | [232.6; 329.4] | 0 | 4 | 996 |
| change-trend-medium | 256 | 1.18 @ 296 | recovered | 320 | 1.250 | [232; 328] | 7 | 437 | 556 |
| change-trend-strong | 256 | 1.39 @ 72 | not_confirmed | — | — | [400; 432] | 0 | 69 | 931 |
| change-ar-weak | 256 | 1.08 @ 8 | recovered | 32 | 0.125 | [24; 360] | 108 | 391 | 501 |
| change-ar-medium | 256 | 1.09 @ 72 | recovered | 280 | 1.094 | [80; 440] | 8 | 797 | 195 |
| change-ar-strong | 256 | 1.07 @ 16 | recovered | 40 | 0.156 | [24; 416] | 53 | 717 | 230 |
| change-seasonality-weak | 256 | 1.30 @ 16 | recovered | 440 | 1.719 | [32; 440] | 0 | 481 | 519 |
| change-seasonality-medium | 256 | 1.41 @ 0 | recovered | 224 | 0.875 | [48; 400] | 0 | 760 | 240 |
| change-seasonality-strong | 256 | 1.51 @ 0 | recovered | 136 | 0.531 | [32; 232] | 0 | 991 | 9 |
| change-frequency-weak | 256 | 1.27 @ 112 | not_confirmed | — | — | [159; 440] | 0 | 116 | 884 |
| change-frequency-medium | 256 | 1.27 @ 48 | not_confirmed | — | — | [64; 280] | 0 | 173 | 827 |
| change-frequency-strong | 256 | 1.52 @ 16 | recovered | 216 | 0.844 | [184; 376] | 0 | 857 | 143 |
| change-sensor_bias-weak | 256 | 1.12 @ 40 | not_confirmed | — | — | [56; 344] | 1 | 370 | 629 |
| change-sensor_bias-medium | 256 | 1.24 @ 0 | not_confirmed | — | — | [16; 440] | 0 | 201 | 799 |
| change-sensor_bias-strong | 256 | 1.71 @ 0 | not_confirmed | — | — | [360; 408] | 0 | 142 | 858 |
| drift-mean-weak ≈ | 256 | 1.15 @ 328 | recovered | 376 | 1.469 | [88; 432] | 0 | 778 | 222 |
| drift-mean-medium ≈ | 256 | 1.24 @ 48 | recovered | 304 | 1.188 | [96; 440] | 0 | 884 | 116 |
| drift-mean-strong ≈ | 256 | 1.48 @ 88 | recovered | 360 | 1.406 | [280; 408] | 0 | 699 | 301 |
| drift-frequency-weak ≈ | 256 | 1.09 @ 144 | not_confirmed | — | — | [176; 320] | 5 | 162 | 833 |
| drift-frequency-medium ≈ | 256 | 1.25 @ 112 | not_confirmed | — | — | [128; 328] | 0 | 140 | 860 |
| drift-frequency-strong ≈ | 256 | 1.42 @ 120 | not_confirmed | — | — | [264; 344] | 0 | 66 | 934 |
| anomaly-spike-weak | 256 | 1.07 @ 0 | recovered | 8 | 0.031 | [8; 296] | 220 | 415 | 365 |
| anomaly-spike-medium | 256 | 1.30 @ 0 | recovered | 424 | 1.656 | [16; 432] | 1 | 481 | 518 |
| anomaly-spike-strong | 256 | 1.61 @ 0 | recovered | 416 | 1.625 | [72; 440] | 0 | 601 | 399 |
| anomaly-freeze-weak | 256 | 1.08 @ 8 | recovered | 16 | 0.062 | [8; 408] | 86 | 399 | 515 |
| anomaly-freeze-medium | 256 | 1.56 @ 40 | not_confirmed | — | — | [185.6; 440] | 0 | 209 | 791 |
| anomaly-freeze-strong | 256 | 1.66 @ 0 | not_confirmed | — | — | [208; 432] | 0 | 277 | 723 |
| none | 256 | 1.10 @ 112 | recovered | 128 | 0.500 | [96; 408] | 231 | 582 | 187 |
| change-mean-weak | 128 | 1.19 @ 216 | not_confirmed | — | — | [232; 432] | 0 | 111 | 889 |
| change-mean-medium | 128 | 1.22 @ 504 | not_confirmed | — | — | [289; 311] | 0 | 6 | 994 |
| change-mean-strong | 128 | 1.42 @ 0 | not_confirmed | — | — | [96.2; 440] | 0 | 162 | 838 |
| change-variance-weak | 128 | 1.15 @ 48 | not_confirmed | — | — | [245.4; 432] | 1 | 80 | 919 |
| change-variance-medium | 128 | 1.19 @ 280 | not_confirmed | — | — | [312; 440] | 0 | 367 | 633 |
| change-variance-strong | 128 | 1.11 @ 216 | recovered | 336 | 2.625 | [80; 440] | 2 | 782 | 216 |
| change-trend-weak | 128 | 1.19 @ 208 | not_confirmed | — | — | — | 0 | 0 | 1000 |
| change-trend-medium | 128 | 1.22 @ 280 | not_confirmed | — | — | [312; 440] | 0 | 280 | 720 |
| change-trend-strong | 128 | 1.35 @ 360 | not_confirmed | — | — | [386.4; 440] | 0 | 47 | 953 |
| change-ar-weak | 128 | 1.27 @ 0 | not_confirmed | — | — | [232; 352] | 0 | 74 | 926 |
| change-ar-medium | 128 | 1.19 @ 24 | not_confirmed | — | — | [346; 440] | 0 | 11 | 989 |
| change-ar-strong | 128 | 1.16 @ 0 | not_confirmed | — | — | [136; 440] | 0 | 109 | 891 |
| change-seasonality-weak | 128 | 1.34 @ 16 | not_confirmed | — | — | [40; 424] | 0 | 174 | 826 |
| change-seasonality-medium | 128 | 1.37 @ 0 | not_confirmed | — | — | [69.2; 408] | 0 | 227 | 773 |
| change-seasonality-strong | 128 | 1.46 @ 8 | recovered | 88 | 0.688 | [48; 432] | 0 | 584 | 416 |
| change-frequency-weak | 128 | 1.17 @ 104 | not_confirmed | — | — | [112; 416] | 0 | 34 | 966 |
| change-frequency-medium | 128 | 1.16 @ 48 | not_confirmed | — | — | [64; 283.2] | 0 | 25 | 975 |
| change-frequency-strong | 128 | 1.27 @ 16 | recovered | 424 | 3.312 | [80; 424] | 0 | 394 | 606 |
| change-sensor_bias-weak | 128 | 1.20 @ 176 | not_confirmed | — | — | [224; 352] | 0 | 366 | 634 |
| change-sensor_bias-medium | 128 | 1.19 @ 0 | not_confirmed | — | — | [185.2; 421.6] | 0 | 7 | 993 |
| change-sensor_bias-strong | 128 | 1.44 @ 0 | not_confirmed | — | — | [190; 366] | 0 | 26 | 974 |
| drift-mean-weak ≈ | 128 | 1.23 @ 328 | not_confirmed | — | — | [368; 440] | 0 | 266 | 734 |
| drift-mean-medium ≈ | 128 | 1.17 @ 48 | not_confirmed | — | — | [102.6; 440] | 0 | 138 | 862 |
| drift-mean-strong ≈ | 128 | 1.30 @ 88 | not_confirmed | — | — | [280; 416] | 0 | 161 | 839 |
| drift-frequency-weak ≈ | 128 | 1.05 @ 432 | not_confirmed | — | — | [144; 440] | 16 | 302 | 682 |
| drift-frequency-medium ≈ | 128 | 1.21 @ 104 | not_confirmed | — | — | [193.8; 262.2] | 0 | 2 | 998 |
| drift-frequency-strong ≈ | 128 | 1.25 @ 48 | not_confirmed | — | — | [141.4; 440] | 0 | 8 | 992 |
| anomaly-spike-weak | 128 | 1.30 @ 0 | recovered | 224 | 1.750 | [16; 424] | 6 | 429 | 565 |
| anomaly-spike-medium | 128 | 1.52 @ 0 | not_confirmed | — | — | [16; 416] | 3 | 447 | 550 |
| anomaly-spike-strong | 128 | 2.36 @ 0 | not_confirmed | — | — | — | 0 | 0 | 1000 |
| anomaly-freeze-weak | 128 | 1.13 @ 256 | not_confirmed | — | — | [36.8; 425.6] | 1 | 73 | 926 |
| anomaly-freeze-medium | 128 | 1.43 @ 48 | not_confirmed | — | — | [416; 440] | 0 | 70 | 930 |
| anomaly-freeze-strong | 128 | 2.56 @ 128 | not_confirmed | — | — | [231.4; 437.4] | 0 | 14 | 986 |
| none | 128 | 1.01 @ 304 | no_degradation | — | — | [120; 440] | 145 | 695 | 160 |
| change-mean-weak | 64 | 1.16 @ 504 | not_confirmed | — | — | [156.8; 304] | 0 | 9 | 991 |
| change-mean-medium | 64 | 1.26 @ 504 | not_confirmed | — | — | — | 0 | 0 | 1000 |
| change-mean-strong | 64 | 1.35 @ 0 | not_confirmed | — | — | [32; 432] | 0 | 67 | 933 |
| change-variance-weak | 64 | 1.12 @ 48 | not_confirmed | — | — | [56; 440] | 11 | 199 | 790 |
| change-variance-medium | 64 | 1.18 @ 272 | recovered | 432 | 6.750 | [296; 440] | 0 | 599 | 401 |
| change-variance-strong | 64 | 1.12 @ 216 | recovered | 328 | 5.125 | [114.6; 440] | 0 | 574 | 426 |
| change-trend-weak | 64 | 1.16 @ 448 | not_confirmed | — | — | — | 0 | 0 | 1000 |
| change-trend-medium | 64 | 1.11 @ 272 | recovered | 328 | 5.125 | [312; 408.4] | 0 | 679 | 321 |
| change-trend-strong | 64 | 1.11 @ 224 | recovered | 384 | 6.000 | [232; 440] | 0 | 415 | 585 |
| change-ar-weak | 64 | 1.31 @ 0 | not_confirmed | — | — | — | 0 | 0 | 1000 |
| change-ar-medium | 64 | 1.17 @ 272 | not_confirmed | — | — | [353; 439] | 0 | 6 | 994 |
| change-ar-strong | 64 | 1.10 @ 0 | not_confirmed | — | — | [364; 440] | 0 | 30 | 970 |
| change-seasonality-weak | 64 | 1.33 @ 16 | not_confirmed | — | — | [168; 416] | 0 | 236 | 764 |
| change-seasonality-medium | 64 | 1.45 @ 0 | not_confirmed | — | — | [224; 424.2] | 0 | 54 | 946 |
| change-seasonality-strong | 64 | 1.30 @ 8 | recovered | 56 | 0.875 | [32; 256] | 0 | 485 | 515 |
| change-frequency-weak | 64 | 1.15 @ 144 | not_confirmed | — | — | [354.8; 440] | 0 | 27 | 973 |
| change-frequency-medium | 64 | 1.12 @ 448 | not_confirmed | — | — | [211.2; 328] | 0 | 28 | 972 |
| change-frequency-strong | 64 | 1.19 @ 8 | not_confirmed | — | — | [40; 440] | 0 | 149 | 851 |
| change-sensor_bias-weak | 64 | 1.06 @ 504 | not_confirmed | — | — | [88; 352] | 0 | 483 | 517 |
| change-sensor_bias-medium | 64 | 1.27 @ 456 | not_confirmed | — | — | — | 0 | 0 | 1000 |
| change-sensor_bias-strong | 64 | 1.34 @ 0 | not_confirmed | — | — | [363; 400] | 0 | 4 | 996 |
| drift-mean-weak ≈ | 64 | 1.22 @ 328 | not_confirmed | — | — | [360; 440] | 0 | 129 | 871 |
| drift-mean-medium ≈ | 64 | 1.12 @ 48 | not_confirmed | — | — | [72; 440] | 0 | 280 | 720 |
| drift-mean-strong ≈ | 64 | 1.27 @ 32 | not_confirmed | — | — | [274; 416] | 0 | 211 | 789 |
| drift-frequency-weak ≈ | 64 | 1.09 @ 128 | not_confirmed | — | — | [320; 432] | 0 | 241 | 759 |
| drift-frequency-medium ≈ | 64 | 1.03 @ 408 | not_confirmed | — | — | [168; 432] | 12 | 143 | 845 |
| drift-frequency-strong ≈ | 64 | 1.16 @ 40 | not_confirmed | — | — | [55.8; 432] | 0 | 40 | 960 |
| anomaly-spike-weak | 64 | 1.29 @ 0 | not_confirmed | — | — | [14.2; 432] | 1 | 312 | 687 |
| anomaly-spike-medium | 64 | 1.69 @ 0 | not_confirmed | — | — | [16; 432] | 0 | 232 | 768 |
| anomaly-spike-strong | 64 | 2.49 @ 0 | not_confirmed | — | — | [374.8; 426.4] | 0 | 35 | 965 |
| anomaly-freeze-weak | 64 | 1.22 @ 72 | not_confirmed | — | — | [308; 424] | 0 | 31 | 969 |
| anomaly-freeze-medium | 64 | 1.79 @ 72 | not_confirmed | — | — | [432; 440] | 0 | 16 | 984 |
| anomaly-freeze-strong | 64 | 2.35 @ 128 | not_confirmed | — | — | [182.4; 440] | 0 | 37 | 963 |
| none | 64 | 1.11 @ 264 | not_confirmed | — | — | [120; 430.8] | 0 | 324 | 676 |

## Чувствительность: без drift

Состав основной оценки не меняется; таблица ниже исключает только приближённый эталон drift.

| L | H | Q | Статус | Ячеек |
| --- | --- | --- | --- | --- |
| 64 | 16 | 90 | not_confirmed | 21 |
| 64 | 16 | 90 | recovered | 7 |
| 64 | 16 | 95 | not_confirmed | 17 |
| 64 | 16 | 95 | recovered | 11 |
| 64 | 64 | 90 | not_confirmed | 23 |
| 64 | 64 | 90 | recovered | 5 |
| 64 | 64 | 95 | not_confirmed | 21 |
| 64 | 64 | 95 | recovered | 7 |
| 128 | 16 | 90 | no_degradation | 1 |
| 128 | 16 | 90 | not_confirmed | 11 |
| 128 | 16 | 90 | recovered | 16 |
| 128 | 16 | 95 | no_degradation | 1 |
| 128 | 16 | 95 | not_confirmed | 11 |
| 128 | 16 | 95 | recovered | 16 |
| 128 | 64 | 90 | no_degradation | 1 |
| 128 | 64 | 90 | not_confirmed | 23 |
| 128 | 64 | 90 | recovered | 4 |
| 128 | 64 | 95 | no_degradation | 1 |
| 128 | 64 | 95 | not_confirmed | 19 |
| 128 | 64 | 95 | recovered | 8 |
| 256 | 16 | 90 | not_confirmed | 18 |
| 256 | 16 | 90 | recovered | 10 |
| 256 | 16 | 95 | not_confirmed | 15 |
| 256 | 16 | 95 | recovered | 13 |
| 256 | 64 | 90 | no_degradation | 1 |
| 256 | 64 | 90 | not_confirmed | 10 |
| 256 | 64 | 90 | recovered | 17 |
| 256 | 64 | 95 | no_degradation | 5 |
| 256 | 64 | 95 | not_confirmed | 8 |
| 256 | 64 | 95 | recovered | 15 |
| 512 | 16 | 90 | not_confirmed | 14 |
| 512 | 16 | 90 | recovered | 14 |
| 512 | 16 | 95 | not_confirmed | 11 |
| 512 | 16 | 95 | recovered | 17 |
| 512 | 64 | 90 | not_confirmed | 18 |
| 512 | 64 | 90 | recovered | 10 |
| 512 | 64 | 95 | no_degradation | 1 |
| 512 | 64 | 95 | not_confirmed | 10 |
| 512 | 64 | 95 | recovered | 17 |

## H2: покрытие 80%-интервала, L=512, H=16

| Набор | Зона d | Coverage [CI95] | MAE × | Ширина × |
| --- | --- | --- | --- | --- |
| все ячейки | все окна | 0.790 [0.786; 0.793] | 1.028 | 1.000 |
| все ячейки | до смены | 0.805 [0.800; 0.810] | 1.002 | 1.001 |
| все ячейки | смена в горизонте | 0.765 [0.749; 0.779] | 1.106 | 0.963 |
| все ячейки | 0–63 | 0.738 [0.726; 0.749] | 1.065 | 0.944 |
| все ячейки | 64–127 | 0.759 [0.747; 0.770] | 1.038 | 0.957 |
| все ячейки | 128–255 | 0.780 [0.772; 0.787] | 1.066 | 1.014 |
| все ячейки | 256–511 | 0.802 [0.798; 0.807] | 1.018 | 1.017 |
| все ячейки | ≥512 | 0.805 [0.800; 0.811] | 1.005 | 1.004 |
| без drift | все окна | 0.793 [0.789; 0.796] | 1.022 | 1.000 |
| без drift | до смены | 0.804 [0.798; 0.810] | 1.001 | 1.001 |
| без drift | смена в горизонте | 0.750 [0.733; 0.767] | 1.141 | 0.963 |
| без drift | 0–63 | 0.734 [0.719; 0.748] | 1.080 | 0.953 |
| без drift | 64–127 | 0.765 [0.753; 0.778] | 1.018 | 0.955 |
| без drift | 128–255 | 0.789 [0.781; 0.796] | 1.048 | 1.015 |
| без drift | 256–511 | 0.807 [0.803; 0.811] | 1.008 | 1.016 |
| без drift | ≥512 | 0.805 [0.800; 0.811] | 1.006 | 1.004 |
| change | все окна | 0.788 [0.784; 0.792] | 1.034 | 1.000 |
| change | до смены | 0.804 [0.797; 0.811] | 1.002 | 1.001 |
| change | смена в горизонте | 0.752 [0.731; 0.772] | 1.075 | 0.952 |
| change | 0–63 | 0.716 [0.699; 0.732] | 1.129 | 0.940 |
| change | 64–127 | 0.746 [0.731; 0.761] | 1.120 | 1.000 |
| change | 128–255 | 0.780 [0.770; 0.788] | 1.064 | 1.010 |
| change | 256–511 | 0.807 [0.802; 0.813] | 1.006 | 1.013 |
| change | ≥512 | 0.804 [0.797; 0.812] | 1.003 | 1.000 |
| drift | все окна | 0.777 [0.766; 0.787] | 1.057 | 0.997 |
| drift | до смены | 0.811 [0.798; 0.824] | 1.006 | 1.002 |
| drift | смена в горизонте | 0.832 [0.809; 0.857] | 0.957 | 0.963 |
| drift | 0–63 | 0.755 [0.731; 0.778] | 0.998 | 0.903 |
| drift | 64–127 | 0.728 [0.696; 0.758] | 1.136 | 0.963 |
| drift | 128–255 | 0.739 [0.716; 0.761] | 1.155 | 1.010 |
| drift | 256–511 | 0.779 [0.764; 0.794] | 1.064 | 1.020 |
| drift | ≥512 | 0.804 [0.789; 0.819] | 1.004 | 1.005 |
| anomaly | все окна | 0.807 [0.801; 0.813] | 0.988 | 1.002 |
| anomaly | до смены | 0.803 [0.792; 0.814] | 1.002 | 1.001 |
| anomaly | смена в горизонте | 0.732 [0.696; 0.766] | 1.437 | 1.002 |
| anomaly | 0–63 | 0.785 [0.760; 0.808] | 0.943 | 0.996 |
| anomaly | 64–127 | 0.834 [0.818; 0.851] | 0.720 | 0.811 |
| anomaly | 128–255 | 0.817 [0.804; 0.830] | 1.002 | 1.031 |
| anomaly | 256–511 | 0.807 [0.799; 0.815] | 1.022 | 1.031 |
| anomaly | ≥512 | 0.806 [0.794; 0.817] | 1.024 | 1.022 |
| none | все окна | 0.805 [0.794; 0.816] | 0.988 | 0.995 |
| none | до смены | 0.801 [0.764; 0.834] | 0.989 | 1.002 |
| none | смена в горизонте | 0.822 [0.747; 0.894] | 1.004 | 0.977 |
| none | 0–63 | 0.819 [0.777; 0.852] | 0.965 | 0.982 |
| none | 64–127 | 0.752 [0.698; 0.797] | 1.094 | 0.981 |
| none | 128–255 | 0.801 [0.777; 0.825] | 0.997 | 1.002 |
| none | 256–511 | 0.806 [0.790; 0.824] | 0.978 | 0.999 |
| none | ≥512 | 0.824 [0.802; 0.845] | 0.962 | 0.988 |

## H2: покрытие 80%-интервала, L=512, H=64

| Набор | Зона d | Coverage [CI95] | MAE × | Ширина × |
| --- | --- | --- | --- | --- |
| все ячейки | все окна | 0.812 [0.806; 0.817] | 1.044 | 1.003 |
| все ячейки | до смены | 0.842 [0.835; 0.848] | 0.999 | 1.001 |
| все ячейки | смена в горизонте | 0.785 [0.774; 0.795] | 1.078 | 0.946 |
| все ячейки | 0–63 | 0.750 [0.735; 0.764] | 1.112 | 0.949 |
| все ячейки | 64–127 | 0.766 [0.751; 0.780] | 1.114 | 0.971 |
| все ячейки | 128–255 | 0.800 [0.790; 0.810] | 1.085 | 1.023 |
| все ячейки | 256–511 | 0.833 [0.826; 0.840] | 1.016 | 1.029 |
| все ячейки | ≥512 | 0.833 [0.826; 0.839] | 1.001 | 1.005 |
| без drift | все окна | 0.818 [0.813; 0.824] | 1.042 | 1.004 |
| без drift | до смены | 0.842 [0.834; 0.849] | 1.000 | 1.001 |
| без drift | смена в горизонте | 0.772 [0.759; 0.784] | 1.120 | 0.944 |
| без drift | 0–63 | 0.750 [0.734; 0.767] | 1.128 | 0.957 |
| без drift | 64–127 | 0.781 [0.766; 0.795] | 1.097 | 0.973 |
| без drift | 128–255 | 0.814 [0.803; 0.824] | 1.070 | 1.024 |
| без drift | 256–511 | 0.842 [0.836; 0.848] | 1.008 | 1.030 |
| без drift | ≥512 | 0.838 [0.831; 0.845] | 1.001 | 1.005 |
| change | все окна | 0.811 [0.804; 0.819] | 1.051 | 1.001 |
| change | до смены | 0.842 [0.834; 0.851] | 1.000 | 1.001 |
| change | смена в горизонте | 0.768 [0.753; 0.783] | 1.093 | 0.927 |
| change | 0–63 | 0.721 [0.703; 0.740] | 1.174 | 0.927 |
| change | 64–127 | 0.766 [0.748; 0.784] | 1.145 | 0.999 |
| change | 128–255 | 0.802 [0.789; 0.815] | 1.095 | 1.022 |
| change | 256–511 | 0.841 [0.832; 0.849] | 1.007 | 1.029 |
| change | ≥512 | 0.834 [0.827; 0.842] | 1.002 | 1.001 |
| drift | все окна | 0.781 [0.765; 0.796] | 1.052 | 0.998 |
| drift | до смены | 0.840 [0.824; 0.856] | 0.995 | 1.003 |
| drift | смена в горизонте | 0.846 [0.830; 0.860] | 0.900 | 0.955 |
| drift | 0–63 | 0.746 [0.716; 0.774] | 1.037 | 0.910 |
| drift | 64–127 | 0.696 [0.660; 0.730] | 1.200 | 0.959 |
| drift | 128–255 | 0.737 [0.707; 0.764] | 1.160 | 1.017 |
| drift | 256–511 | 0.787 [0.765; 0.808] | 1.053 | 1.028 |
| drift | ≥512 | 0.807 [0.788; 0.825] | 0.999 | 1.003 |
| anomaly | все окна | 0.838 [0.831; 0.846] | 1.023 | 1.019 |
| anomaly | до смены | 0.840 [0.825; 0.854] | 1.000 | 1.001 |
| anomaly | смена в горизонте | 0.770 [0.743; 0.796] | 1.248 | 0.999 |
| anomaly | 0–63 | 0.837 [0.817; 0.856] | 1.003 | 1.069 |
| anomaly | 64–127 | 0.829 [0.803; 0.851] | 0.951 | 0.889 |
| anomaly | 128–255 | 0.848 [0.835; 0.861] | 1.005 | 1.039 |
| anomaly | 256–511 | 0.847 [0.836; 0.857] | 1.018 | 1.039 |
| anomaly | ≥512 | 0.847 [0.834; 0.860] | 1.008 | 1.024 |
| none | все окна | 0.848 [0.829; 0.865] | 0.978 | 0.986 |
| none | до смены | 0.843 [0.802; 0.884] | 1.000 | 1.003 |
| none | смена в горизонте | 0.860 [0.827; 0.889] | 0.976 | 0.986 |
| none | 0–63 | 0.838 [0.798; 0.868] | 0.987 | 0.970 |
| none | 64–127 | 0.813 [0.760; 0.859] | 1.041 | 0.957 |
| none | 128–255 | 0.849 [0.812; 0.880] | 0.966 | 0.993 |
| none | 256–511 | 0.849 [0.826; 0.872] | 0.981 | 0.988 |
| none | ≥512 | 0.862 [0.831; 0.894] | 0.943 | 0.987 |

## Аудит и файлы

`reference_audit.csv`: эталоны и число окон по каждому ряду/L/H/метрике. `exclusion_audit.csv`: объём исключений. `thresholds.csv`: сырой и ограниченный порог.
`curves.csv.gz`: описательные кривые/CI и n_valid; `d_star.csv`: все статусы, частоты бутстрепа, условные CI и d*/L. `coverage_zones.csv`, `coverage_zones_no_drift.csv`: H2.
Исключено отношений (по метрикам суммарно): 0. Окна перекрываются; это не число независимых наблюдений.
Время расчёта: 47.0 с (только кеш, без инференса).
