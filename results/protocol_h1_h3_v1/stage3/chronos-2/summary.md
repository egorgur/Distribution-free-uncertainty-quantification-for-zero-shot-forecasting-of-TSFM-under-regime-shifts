# Этап 3 — amazon/chronos-2, протокол v1

Дата: 2026-11-12 21:32. Протокол: `h1_h3_v1`, SHA256 `22bae05a71811c6cf7c17b2bac1d3a80ff106f010f2eff3cabd612bb816e8dc1`.
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
| 64 | 16 | 95 | 1.045378 | 1.045378 |
| 64 | 16 | 90 | 1.003847 | 1.003847 |
| 64 | 64 | 95 | 1.031255 | 1.031255 |
| 64 | 64 | 90 | 0.996994 | 1.000000 |
| 128 | 16 | 95 | 1.113646 | 1.113646 |
| 128 | 16 | 90 | 1.089085 | 1.089085 |
| 128 | 64 | 95 | 1.046727 | 1.046727 |
| 128 | 64 | 90 | 1.018937 | 1.018937 |
| 256 | 16 | 95 | 1.095328 | 1.095328 |
| 256 | 16 | 90 | 1.069095 | 1.069095 |
| 256 | 64 | 95 | 1.090943 | 1.090943 |
| 256 | 64 | 90 | 1.046077 | 1.046077 |
| 512 | 16 | 95 | 1.095491 | 1.095491 |
| 512 | 16 | 90 | 1.073855 | 1.073855 |
| 512 | 64 | 95 | 1.020736 | 1.020736 |
| 512 | 64 | 90 | 1.009800 | 1.009800 |

## H1: H=16, Q95 (основной)

CI в таблице — только по recovered-повторам. Drift помечен `≈`.

| Ячейка | L | Сглаженный пик @ d | Статус | d* | d*/L | Условный CI95 d* | boot: нет дегр. | boot: recovered | boot: не подтв. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| change-mean-weak | 512 | 1.14 @ 504 | not_confirmed | — | — | [128; 432] | 1 | 409 | 590 |
| change-mean-medium | 512 | 1.38 @ 0 | recovered | 432 | 0.844 | [296; 440] | 0 | 483 | 517 |
| change-mean-strong | 512 | 1.90 @ 0 | not_confirmed | — | — | [56; 400] | 0 | 401 | 599 |
| change-variance-weak | 512 | 1.22 @ 280 | not_confirmed | — | — | [48; 304] | 1 | 151 | 848 |
| change-variance-medium | 512 | 1.17 @ 264 | recovered | 336 | 0.656 | [272; 440] | 0 | 620 | 380 |
| change-variance-strong | 512 | 1.15 @ 504 | not_confirmed | — | — | [88; 336] | 1 | 611 | 388 |
| change-trend-weak | 512 | 1.25 @ 112 | not_confirmed | — | — | [71.4; 392.6] | 0 | 38 | 962 |
| change-trend-medium | 512 | 1.42 @ 104 | recovered | 328 | 0.641 | [160; 344] | 0 | 818 | 182 |
| change-trend-strong | 512 | 1.70 @ 72 | not_confirmed | — | — | [256; 432] | 0 | 454 | 546 |
| change-ar-weak | 512 | 1.14 @ 192 | recovered | 208 | 0.406 | [40; 360] | 13 | 789 | 198 |
| change-ar-medium | 512 | 1.15 @ 312 | recovered | 328 | 0.641 | [120; 440] | 9 | 428 | 563 |
| change-ar-strong | 512 | 1.09 @ 24 | no_degradation | — | — | [32; 344] | 199 | 527 | 274 |
| change-seasonality-weak | 512 | 1.47 @ 0 | recovered | 248 | 0.484 | [24; 416] | 0 | 552 | 448 |
| change-seasonality-medium | 512 | 1.30 @ 24 | recovered | 208 | 0.406 | [104; 392] | 0 | 588 | 412 |
| change-seasonality-strong | 512 | 1.72 @ 0 | recovered | 200 | 0.391 | [32; 408] | 0 | 851 | 149 |
| change-frequency-weak | 512 | 1.34 @ 144 | recovered | 176 | 0.344 | [160; 320] | 0 | 595 | 405 |
| change-frequency-medium | 512 | 1.91 @ 40 | recovered | 432 | 0.844 | [252.8; 440] | 0 | 385 | 615 |
| change-frequency-strong | 512 | 2.38 @ 16 | recovered | 368 | 0.719 | [216; 416] | 0 | 879 | 121 |
| change-sensor_bias-weak | 512 | 1.16 @ 80 | recovered | 176 | 0.344 | [80; 392] | 2 | 789 | 209 |
| change-sensor_bias-medium | 512 | 1.35 @ 0 | not_confirmed | — | — | [206; 421.8] | 0 | 12 | 988 |
| change-sensor_bias-strong | 512 | 2.47 @ 0 | not_confirmed | — | — | [384; 416.6] | 0 | 38 | 962 |
| drift-mean-weak ≈ | 512 | 1.26 @ 368 | recovered | 392 | 0.766 | [376; 440] | 0 | 509 | 491 |
| drift-mean-medium ≈ | 512 | 1.38 @ 48 | recovered | 184 | 0.359 | [88; 432] | 0 | 740 | 260 |
| drift-mean-strong ≈ | 512 | 1.43 @ 88 | recovered | 360 | 0.703 | [184; 408] | 0 | 753 | 247 |
| drift-frequency-weak ≈ | 512 | 1.50 @ 144 | not_confirmed | — | — | [344; 436.4] | 0 | 99 | 901 |
| drift-frequency-medium ≈ | 512 | 1.55 @ 112 | not_confirmed | — | — | [328; 440] | 0 | 21 | 979 |
| drift-frequency-strong ≈ | 512 | 1.76 @ 56 | recovered | 320 | 0.625 | [288; 400] | 0 | 554 | 446 |
| anomaly-spike-weak | 512 | 1.22 @ 184 | recovered | 280 | 0.547 | [192; 440] | 0 | 597 | 403 |
| anomaly-spike-medium | 512 | 1.65 @ 0 | recovered | 120 | 0.234 | [16; 440] | 1 | 664 | 335 |
| anomaly-spike-strong | 512 | 3.15 @ 0 | not_confirmed | — | — | [411.2; 440] | 0 | 257 | 743 |
| anomaly-freeze-weak | 512 | 1.14 @ 8 | recovered | 16 | 0.031 | [8; 376] | 53 | 512 | 435 |
| anomaly-freeze-medium | 512 | 1.61 @ 0 | not_confirmed | — | — | [8; 440] | 0 | 219 | 781 |
| anomaly-freeze-strong | 512 | 1.33 @ 0 | not_confirmed | — | — | [8; 432] | 1 | 128 | 871 |
| none | 512 | 1.12 @ 80 | recovered | 136 | 0.266 | [88; 408.4] | 113 | 479 | 408 |
| change-mean-weak | 256 | 1.18 @ 216 | not_confirmed | — | — | [144; 440] | 0 | 304 | 696 |
| change-mean-medium | 256 | 1.19 @ 8 | recovered | 32 | 0.125 | [16; 424] | 4 | 895 | 101 |
| change-mean-strong | 256 | 1.68 @ 0 | recovered | 384 | 1.500 | [128; 392] | 0 | 525 | 475 |
| change-variance-weak | 256 | 1.17 @ 40 | recovered | 296 | 1.156 | [48; 312] | 13 | 386 | 601 |
| change-variance-medium | 256 | 1.20 @ 264 | recovered | 336 | 1.312 | [272; 440] | 0 | 516 | 484 |
| change-variance-strong | 256 | 1.14 @ 504 | not_confirmed | — | — | [104; 408] | 3 | 665 | 332 |
| change-trend-weak | 256 | 1.26 @ 448 | not_confirmed | — | — | [272; 272] | 0 | 1 | 999 |
| change-trend-medium | 256 | 1.21 @ 464 | not_confirmed | — | — | [96; 360] | 0 | 406 | 594 |
| change-trend-strong | 256 | 1.38 @ 72 | recovered | 288 | 1.125 | [142; 416] | 0 | 831 | 169 |
| change-ar-weak | 256 | 1.14 @ 200 | recovered | 216 | 0.844 | [40; 368] | 3 | 572 | 425 |
| change-ar-medium | 256 | 1.17 @ 96 | not_confirmed | — | — | [96; 440] | 6 | 438 | 556 |
| change-ar-strong | 256 | 1.12 @ 24 | recovered | 96 | 0.375 | [24; 334.8] | 11 | 647 | 342 |
| change-seasonality-weak | 256 | 1.56 @ 0 | recovered | 40 | 0.156 | [32; 408] | 0 | 719 | 281 |
| change-seasonality-medium | 256 | 1.31 @ 0 | recovered | 240 | 0.938 | [48; 392] | 0 | 629 | 371 |
| change-seasonality-strong | 256 | 1.50 @ 0 | recovered | 184 | 0.719 | [32; 216] | 0 | 982 | 18 |
| change-frequency-weak | 256 | 1.19 @ 144 | not_confirmed | — | — | [152; 320] | 0 | 234 | 766 |
| change-frequency-medium | 256 | 1.69 @ 24 | not_confirmed | — | — | [143.2; 440] | 0 | 277 | 723 |
| change-frequency-strong | 256 | 2.12 @ 16 | recovered | 168 | 0.656 | [136; 400] | 0 | 665 | 335 |
| change-sensor_bias-weak | 256 | 1.15 @ 168 | recovered | 184 | 0.719 | [64; 386] | 2 | 791 | 207 |
| change-sensor_bias-medium | 256 | 1.31 @ 0 | not_confirmed | — | — | [31.4; 424.6] | 0 | 38 | 962 |
| change-sensor_bias-strong | 256 | 2.10 @ 0 | not_confirmed | — | — | [176; 416] | 0 | 220 | 780 |
| drift-mean-weak ≈ | 256 | 1.27 @ 368 | recovered | 392 | 1.531 | [376; 440] | 0 | 648 | 352 |
| drift-mean-medium ≈ | 256 | 1.32 @ 48 | recovered | 288 | 1.125 | [79.8; 432] | 0 | 760 | 240 |
| drift-mean-strong ≈ | 256 | 1.30 @ 88 | recovered | 360 | 1.406 | [176; 416] | 0 | 768 | 232 |
| drift-frequency-weak ≈ | 256 | 1.23 @ 144 | recovered | 176 | 0.688 | [160; 432] | 0 | 685 | 315 |
| drift-frequency-medium ≈ | 256 | 1.46 @ 120 | not_confirmed | — | — | [160; 440] | 0 | 146 | 854 |
| drift-frequency-strong ≈ | 256 | 1.67 @ 56 | recovered | 272 | 1.062 | [208; 384] | 0 | 675 | 325 |
| anomaly-spike-weak | 256 | 1.22 @ 184 | recovered | 280 | 1.094 | [48; 416.2] | 8 | 560 | 432 |
| anomaly-spike-medium | 256 | 1.65 @ 0 | recovered | 120 | 0.469 | [16; 440] | 0 | 635 | 365 |
| anomaly-spike-strong | 256 | 3.12 @ 0 | recovered | 440 | 1.719 | [344; 440] | 0 | 482 | 518 |
| anomaly-freeze-weak | 256 | 1.19 @ 8 | recovered | 32 | 0.125 | [16; 424] | 27 | 577 | 396 |
| anomaly-freeze-medium | 256 | 1.28 @ 56 | not_confirmed | — | — | [296; 440] | 0 | 300 | 700 |
| anomaly-freeze-strong | 256 | 1.27 @ 128 | not_confirmed | — | — | [176.4; 424] | 0 | 203 | 797 |
| none | 256 | 1.16 @ 168 | recovered | 176 | 0.688 | [112; 364.2] | 64 | 660 | 276 |
| change-mean-weak | 128 | 1.16 @ 416 | recovered | 432 | 3.375 | [88; 432] | 5 | 597 | 398 |
| change-mean-medium | 128 | 1.13 @ 288 | recovered | 296 | 2.312 | [8; 408] | 72 | 822 | 106 |
| change-mean-strong | 128 | 1.40 @ 0 | recovered | 136 | 1.062 | [16; 432] | 0 | 689 | 311 |
| change-variance-weak | 128 | 1.19 @ 40 | recovered | 112 | 0.875 | [48; 320] | 68 | 642 | 290 |
| change-variance-medium | 128 | 1.23 @ 264 | recovered | 304 | 2.375 | [144; 432] | 0 | 724 | 276 |
| change-variance-strong | 128 | 1.13 @ 208 | recovered | 216 | 1.688 | [80; 256] | 75 | 729 | 196 |
| change-trend-weak | 128 | 1.20 @ 216 | not_confirmed | — | — | [232; 342.8] | 0 | 62 | 938 |
| change-trend-medium | 128 | 1.32 @ 224 | recovered | 360 | 2.812 | [360; 440] | 0 | 431 | 569 |
| change-trend-strong | 128 | 1.26 @ 72 | recovered | 408 | 3.188 | [224; 432] | 0 | 663 | 337 |
| change-ar-weak | 128 | 1.16 @ 200 | not_confirmed | — | — | [32; 368] | 5 | 368 | 627 |
| change-ar-medium | 128 | 1.22 @ 8 | not_confirmed | — | — | [16; 440] | 3 | 416 | 581 |
| change-ar-strong | 128 | 1.19 @ 24 | recovered | 40 | 0.312 | [32; 272] | 0 | 831 | 169 |
| change-seasonality-weak | 128 | 1.42 @ 0 | recovered | 40 | 0.312 | [32; 373] | 0 | 936 | 64 |
| change-seasonality-medium | 128 | 1.29 @ 0 | recovered | 120 | 0.938 | [72; 424] | 0 | 745 | 255 |
| change-seasonality-strong | 128 | 1.50 @ 0 | recovered | 48 | 0.375 | [24; 216] | 0 | 991 | 9 |
| change-frequency-weak | 128 | 1.18 @ 272 | recovered | 288 | 2.250 | [56; 325.6] | 4 | 277 | 719 |
| change-frequency-medium | 128 | 1.44 @ 16 | recovered | 112 | 0.875 | [40; 440] | 0 | 629 | 371 |
| change-frequency-strong | 128 | 1.96 @ 16 | recovered | 88 | 0.688 | [64; 376] | 0 | 981 | 19 |
| change-sensor_bias-weak | 128 | 1.12 @ 168 | recovered | 176 | 1.375 | [56; 376] | 187 | 758 | 55 |
| change-sensor_bias-medium | 128 | 1.25 @ 0 | not_confirmed | — | — | [16; 408] | 1 | 63 | 936 |
| change-sensor_bias-strong | 128 | 1.74 @ 0 | recovered | 112 | 0.875 | [104; 392] | 0 | 295 | 705 |
| drift-mean-weak ≈ | 128 | 1.29 @ 368 | recovered | 392 | 3.062 | [376; 440] | 0 | 803 | 197 |
| drift-mean-medium ≈ | 128 | 1.20 @ 64 | recovered | 88 | 0.688 | [72; 440] | 0 | 919 | 81 |
| drift-mean-strong ≈ | 128 | 1.25 @ 344 | recovered | 368 | 2.875 | [112; 440] | 1 | 911 | 88 |
| drift-frequency-weak ≈ | 128 | 1.17 @ 64 | recovered | 160 | 1.250 | [72; 440] | 2 | 468 | 530 |
| drift-frequency-medium ≈ | 128 | 1.20 @ 440 | not_confirmed | — | — | [104; 384] | 0 | 233 | 767 |
| drift-frequency-strong ≈ | 128 | 1.69 @ 80 | not_confirmed | — | — | [128; 400] | 0 | 340 | 660 |
| anomaly-spike-weak | 128 | 1.22 @ 184 | recovered | 280 | 2.188 | [8; 424] | 27 | 555 | 418 |
| anomaly-spike-medium | 128 | 1.82 @ 0 | recovered | 128 | 1.000 | [16; 440] | 0 | 414 | 586 |
| anomaly-spike-strong | 128 | 3.11 @ 0 | recovered | 352 | 2.750 | [200; 440] | 0 | 620 | 380 |
| anomaly-freeze-weak | 128 | 1.26 @ 24 | recovered | 160 | 1.250 | [32; 384] | 5 | 552 | 443 |
| anomaly-freeze-medium | 128 | 1.22 @ 104 | recovered | 112 | 0.875 | [64; 440] | 0 | 740 | 260 |
| anomaly-freeze-strong | 128 | 2.00 @ 136 | not_confirmed | — | — | [168; 440] | 0 | 236 | 764 |
| none | 128 | 1.12 @ 464 | not_confirmed | — | — | [8; 424] | 60 | 491 | 449 |
| change-mean-weak | 64 | 1.32 @ 232 | not_confirmed | — | — | [240; 432] | 0 | 76 | 924 |
| change-mean-medium | 64 | 1.22 @ 0 | not_confirmed | — | — | [216; 432] | 0 | 315 | 685 |
| change-mean-strong | 64 | 1.31 @ 0 | not_confirmed | — | — | [128; 432] | 0 | 328 | 672 |
| change-variance-weak | 64 | 1.22 @ 40 | not_confirmed | — | — | [48; 432] | 0 | 163 | 837 |
| change-variance-medium | 64 | 1.11 @ 416 | recovered | 424 | 6.625 | [92.4; 440] | 0 | 392 | 608 |
| change-variance-strong | 64 | 1.21 @ 208 | recovered | 232 | 3.625 | [120; 416] | 1 | 440 | 559 |
| change-trend-weak | 64 | 1.28 @ 376 | not_confirmed | — | — | [392; 437.6] | 0 | 3 | 997 |
| change-trend-medium | 64 | 1.22 @ 56 | not_confirmed | — | — | [272; 440] | 0 | 316 | 684 |
| change-trend-strong | 64 | 1.20 @ 216 | recovered | 416 | 6.500 | [328; 424] | 0 | 536 | 464 |
| change-ar-weak | 64 | 1.25 @ 0 | not_confirmed | — | — | [234.8; 352] | 0 | 55 | 945 |
| change-ar-medium | 64 | 1.18 @ 0 | not_confirmed | — | — | [28; 440] | 0 | 231 | 769 |
| change-ar-strong | 64 | 1.21 @ 16 | not_confirmed | — | — | [40; 416] | 0 | 331 | 669 |
| change-seasonality-weak | 64 | 1.34 @ 16 | recovered | 200 | 3.125 | [32; 424] | 0 | 450 | 550 |
| change-seasonality-medium | 64 | 1.37 @ 0 | not_confirmed | — | — | [24; 408] | 0 | 335 | 665 |
| change-seasonality-strong | 64 | 1.58 @ 0 | recovered | 40 | 0.625 | [32; 256] | 0 | 837 | 163 |
| change-frequency-weak | 64 | 1.10 @ 136 | recovered | 200 | 3.125 | [144; 320] | 2 | 265 | 733 |
| change-frequency-medium | 64 | 1.33 @ 24 | not_confirmed | — | — | — | 0 | 0 | 1000 |
| change-frequency-strong | 64 | 1.61 @ 16 | recovered | 104 | 1.625 | [40; 416] | 0 | 852 | 148 |
| change-sensor_bias-weak | 64 | 1.22 @ 64 | recovered | 256 | 4.000 | [88; 408] | 2 | 492 | 506 |
| change-sensor_bias-medium | 64 | 1.20 @ 504 | not_confirmed | — | — | — | 0 | 0 | 1000 |
| change-sensor_bias-strong | 64 | 1.26 @ 40 | recovered | 392 | 6.125 | [184; 432] | 0 | 412 | 588 |
| drift-mean-weak ≈ | 64 | 1.27 @ 368 | recovered | 400 | 6.250 | [384; 440] | 0 | 544 | 456 |
| drift-mean-medium ≈ | 64 | 1.24 @ 64 | recovered | 184 | 2.875 | [88; 440] | 0 | 628 | 372 |
| drift-mean-strong ≈ | 64 | 1.19 @ 344 | recovered | 368 | 5.750 | [281.6; 440] | 0 | 937 | 63 |
| drift-frequency-weak ≈ | 64 | 1.22 @ 144 | not_confirmed | — | — | [168; 434] | 0 | 31 | 969 |
| drift-frequency-medium ≈ | 64 | 1.16 @ 256 | not_confirmed | — | — | [8; 432] | 7 | 171 | 822 |
| drift-frequency-strong ≈ | 64 | 1.33 @ 72 | recovered | 392 | 6.125 | [112; 440] | 0 | 262 | 738 |
| anomaly-spike-weak | 64 | 1.24 @ 0 | recovered | 440 | 6.875 | [8; 440] | 1 | 433 | 566 |
| anomaly-spike-medium | 64 | 1.73 @ 0 | not_confirmed | — | — | [128; 440] | 0 | 281 | 719 |
| anomaly-spike-strong | 64 | 2.67 @ 0 | not_confirmed | — | — | [192; 440] | 0 | 308 | 692 |
| anomaly-freeze-weak | 64 | 1.21 @ 24 | not_confirmed | — | — | [40; 423.8] | 0 | 82 | 918 |
| anomaly-freeze-medium | 64 | 1.56 @ 56 | recovered | 424 | 6.625 | [216; 440] | 0 | 451 | 549 |
| anomaly-freeze-strong | 64 | 1.88 @ 144 | not_confirmed | — | — | [392; 440] | 0 | 41 | 959 |
| none | 64 | 1.19 @ 112 | not_confirmed | — | — | [96; 424] | 3 | 231 | 766 |

## H1: H=16, Q90 (чувствительность)

CI в таблице — только по recovered-повторам. Drift помечен `≈`.

| Ячейка | L | Сглаженный пик @ d | Статус | d* | d*/L | Условный CI95 d* | boot: нет дегр. | boot: recovered | boot: не подтв. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| change-mean-weak | 512 | 1.14 @ 504 | not_confirmed | — | — | [136; 440] | 0 | 317 | 683 |
| change-mean-medium | 512 | 1.38 @ 0 | not_confirmed | — | — | [408; 440] | 0 | 327 | 673 |
| change-mean-strong | 512 | 1.90 @ 0 | not_confirmed | — | — | [88; 400] | 0 | 273 | 727 |
| change-variance-weak | 512 | 1.22 @ 280 | not_confirmed | — | — | [48; 304] | 1 | 67 | 932 |
| change-variance-medium | 512 | 1.17 @ 264 | recovered | 344 | 0.672 | [272; 440] | 0 | 482 | 518 |
| change-variance-strong | 512 | 1.15 @ 504 | not_confirmed | — | — | [104; 416] | 0 | 490 | 510 |
| change-trend-weak | 512 | 1.25 @ 112 | not_confirmed | — | — | [241.2; 392] | 0 | 7 | 993 |
| change-trend-medium | 512 | 1.42 @ 104 | recovered | 328 | 0.641 | [200; 352] | 0 | 733 | 267 |
| change-trend-strong | 512 | 1.70 @ 72 | not_confirmed | — | — | [296; 432] | 0 | 365 | 635 |
| change-ar-weak | 512 | 1.14 @ 192 | recovered | 216 | 0.422 | [104; 360] | 0 | 688 | 312 |
| change-ar-medium | 512 | 1.15 @ 312 | not_confirmed | — | — | [160; 440] | 1 | 251 | 748 |
| change-ar-strong | 512 | 1.09 @ 24 | recovered | 32 | 0.062 | [32; 353.6] | 54 | 553 | 393 |
| change-seasonality-weak | 512 | 1.47 @ 0 | recovered | 248 | 0.484 | [32; 416] | 0 | 453 | 547 |
| change-seasonality-medium | 512 | 1.30 @ 24 | recovered | 344 | 0.672 | [112; 392] | 0 | 421 | 579 |
| change-seasonality-strong | 512 | 1.72 @ 0 | recovered | 208 | 0.406 | [32; 408] | 0 | 758 | 242 |
| change-frequency-weak | 512 | 1.34 @ 144 | recovered | 280 | 0.547 | [160; 367] | 0 | 458 | 542 |
| change-frequency-medium | 512 | 1.91 @ 40 | not_confirmed | — | — | [272; 440] | 0 | 289 | 711 |
| change-frequency-strong | 512 | 2.38 @ 16 | recovered | 368 | 0.719 | [224; 416] | 0 | 789 | 211 |
| change-sensor_bias-weak | 512 | 1.16 @ 80 | recovered | 176 | 0.344 | [88; 392] | 0 | 641 | 359 |
| change-sensor_bias-medium | 512 | 1.35 @ 0 | not_confirmed | — | — | [272; 272] | 0 | 1 | 999 |
| change-sensor_bias-strong | 512 | 2.47 @ 0 | not_confirmed | — | — | [384; 415.6] | 0 | 15 | 985 |
| drift-mean-weak ≈ | 512 | 1.26 @ 368 | recovered | 432 | 0.844 | [376; 440] | 0 | 392 | 608 |
| drift-mean-medium ≈ | 512 | 1.38 @ 48 | recovered | 192 | 0.375 | [96; 440] | 0 | 627 | 373 |
| drift-mean-strong ≈ | 512 | 1.43 @ 88 | recovered | 368 | 0.719 | [184; 408] | 0 | 641 | 359 |
| drift-frequency-weak ≈ | 512 | 1.50 @ 144 | not_confirmed | — | — | [352; 432] | 0 | 51 | 949 |
| drift-frequency-medium ≈ | 512 | 1.55 @ 112 | not_confirmed | — | — | [385.2; 430.8] | 0 | 2 | 998 |
| drift-frequency-strong ≈ | 512 | 1.76 @ 56 | not_confirmed | — | — | [288; 400] | 0 | 321 | 679 |
| anomaly-spike-weak | 512 | 1.22 @ 184 | recovered | 288 | 0.562 | [200; 440] | 0 | 469 | 531 |
| anomaly-spike-medium | 512 | 1.65 @ 0 | recovered | 120 | 0.234 | [16; 440] | 0 | 568 | 432 |
| anomaly-spike-strong | 512 | 3.15 @ 0 | not_confirmed | — | — | [416; 440] | 0 | 141 | 859 |
| anomaly-freeze-weak | 512 | 1.14 @ 8 | recovered | 16 | 0.031 | [8; 424] | 18 | 441 | 541 |
| anomaly-freeze-medium | 512 | 1.61 @ 0 | not_confirmed | — | — | [18.8; 440] | 0 | 130 | 870 |
| anomaly-freeze-strong | 512 | 1.33 @ 0 | not_confirmed | — | — | [222.6; 432] | 0 | 54 | 946 |
| none | 512 | 1.12 @ 80 | recovered | 136 | 0.266 | [88; 416] | 74 | 425 | 501 |
| change-mean-weak | 256 | 1.18 @ 216 | not_confirmed | — | — | [144; 440] | 0 | 208 | 792 |
| change-mean-medium | 256 | 1.19 @ 8 | recovered | 304 | 1.188 | [24; 432] | 1 | 793 | 206 |
| change-mean-strong | 256 | 1.68 @ 0 | not_confirmed | — | — | [136; 392] | 0 | 328 | 672 |
| change-variance-weak | 256 | 1.17 @ 40 | not_confirmed | — | — | [107.2; 312.6] | 6 | 238 | 756 |
| change-variance-medium | 256 | 1.20 @ 264 | recovered | 432 | 1.688 | [288; 440] | 0 | 265 | 735 |
| change-variance-strong | 256 | 1.14 @ 504 | not_confirmed | — | — | [104; 416] | 1 | 607 | 392 |
| change-trend-weak | 256 | 1.26 @ 448 | not_confirmed | — | — | [272; 272] | 0 | 1 | 999 |
| change-trend-medium | 256 | 1.21 @ 464 | not_confirmed | — | — | [112; 360] | 0 | 254 | 746 |
| change-trend-strong | 256 | 1.38 @ 72 | recovered | 368 | 1.438 | [152; 424] | 0 | 765 | 235 |
| change-ar-weak | 256 | 1.14 @ 200 | not_confirmed | — | — | [216; 368] | 0 | 419 | 581 |
| change-ar-medium | 256 | 1.17 @ 96 | not_confirmed | — | — | [123.2; 440] | 0 | 233 | 767 |
| change-ar-strong | 256 | 1.12 @ 24 | recovered | 104 | 0.406 | [32; 336] | 0 | 498 | 502 |
| change-seasonality-weak | 256 | 1.56 @ 0 | recovered | 192 | 0.750 | [32; 408] | 0 | 584 | 416 |
| change-seasonality-medium | 256 | 1.31 @ 0 | recovered | 312 | 1.219 | [104; 416] | 0 | 434 | 566 |
| change-seasonality-strong | 256 | 1.50 @ 0 | recovered | 184 | 0.719 | [32; 240] | 0 | 957 | 43 |
| change-frequency-weak | 256 | 1.19 @ 144 | not_confirmed | — | — | [160.8; 381.2] | 0 | 83 | 917 |
| change-frequency-medium | 256 | 1.69 @ 24 | not_confirmed | — | — | [152; 440] | 0 | 131 | 869 |
| change-frequency-strong | 256 | 2.12 @ 16 | recovered | 304 | 1.188 | [136; 400] | 0 | 506 | 494 |
| change-sensor_bias-weak | 256 | 1.15 @ 168 | recovered | 184 | 0.719 | [88; 427] | 0 | 586 | 414 |
| change-sensor_bias-medium | 256 | 1.31 @ 0 | not_confirmed | — | — | [184; 424] | 0 | 12 | 988 |
| change-sensor_bias-strong | 256 | 2.10 @ 0 | not_confirmed | — | — | [214.2; 417.8] | 0 | 112 | 888 |
| drift-mean-weak ≈ | 256 | 1.27 @ 368 | recovered | 392 | 1.531 | [376; 440] | 0 | 483 | 517 |
| drift-mean-medium ≈ | 256 | 1.32 @ 48 | recovered | 288 | 1.125 | [96; 432] | 0 | 619 | 381 |
| drift-mean-strong ≈ | 256 | 1.30 @ 88 | not_confirmed | — | — | [240; 432] | 0 | 592 | 408 |
| drift-frequency-weak ≈ | 256 | 1.23 @ 144 | recovered | 192 | 0.750 | [168; 432] | 0 | 526 | 474 |
| drift-frequency-medium ≈ | 256 | 1.46 @ 120 | not_confirmed | — | — | [176; 440] | 0 | 79 | 921 |
| drift-frequency-strong ≈ | 256 | 1.67 @ 56 | recovered | 280 | 1.094 | [208; 392] | 0 | 514 | 486 |
| anomaly-spike-weak | 256 | 1.22 @ 184 | not_confirmed | — | — | [57; 424] | 4 | 446 | 550 |
| anomaly-spike-medium | 256 | 1.65 @ 0 | recovered | 128 | 0.500 | [16; 440] | 0 | 505 | 495 |
| anomaly-spike-strong | 256 | 3.12 @ 0 | recovered | 440 | 1.719 | [392; 440] | 0 | 337 | 663 |
| anomaly-freeze-weak | 256 | 1.19 @ 8 | recovered | 48 | 0.188 | [16; 424] | 10 | 482 | 508 |
| anomaly-freeze-medium | 256 | 1.28 @ 56 | not_confirmed | — | — | [304; 440] | 0 | 159 | 841 |
| anomaly-freeze-strong | 256 | 1.27 @ 128 | not_confirmed | — | — | [336; 424] | 0 | 102 | 898 |
| none | 256 | 1.16 @ 168 | recovered | 192 | 0.750 | [115.2; 404.8] | 28 | 577 | 395 |
| change-mean-weak | 128 | 1.16 @ 416 | recovered | 432 | 3.375 | [96; 436.8] | 0 | 457 | 543 |
| change-mean-medium | 128 | 1.13 @ 288 | recovered | 304 | 2.375 | [16; 417.8] | 42 | 792 | 166 |
| change-mean-strong | 128 | 1.40 @ 0 | recovered | 144 | 1.125 | [55.6; 432] | 0 | 559 | 441 |
| change-variance-weak | 128 | 1.19 @ 40 | recovered | 120 | 0.938 | [48; 344.4] | 19 | 519 | 462 |
| change-variance-medium | 128 | 1.23 @ 264 | recovered | 304 | 2.375 | [272; 437.8] | 0 | 492 | 508 |
| change-variance-strong | 128 | 1.13 @ 208 | recovered | 216 | 1.688 | [80; 280] | 20 | 741 | 239 |
| change-trend-weak | 128 | 1.20 @ 216 | not_confirmed | — | — | [236.2; 379.8] | 0 | 22 | 978 |
| change-trend-medium | 128 | 1.32 @ 224 | not_confirmed | — | — | [360; 440] | 0 | 304 | 696 |
| change-trend-strong | 128 | 1.26 @ 72 | recovered | 416 | 3.250 | [288; 432] | 0 | 568 | 432 |
| change-ar-weak | 128 | 1.16 @ 200 | not_confirmed | — | — | [101.4; 368] | 0 | 268 | 732 |
| change-ar-medium | 128 | 1.22 @ 8 | not_confirmed | — | — | [32; 440] | 0 | 269 | 731 |
| change-ar-strong | 128 | 1.19 @ 24 | recovered | 40 | 0.312 | [32; 376] | 0 | 710 | 290 |
| change-seasonality-weak | 128 | 1.42 @ 0 | recovered | 40 | 0.312 | [32; 408] | 0 | 892 | 108 |
| change-seasonality-medium | 128 | 1.29 @ 0 | recovered | 120 | 0.938 | [112; 432] | 0 | 626 | 374 |
| change-seasonality-strong | 128 | 1.50 @ 0 | recovered | 48 | 0.375 | [24; 232] | 0 | 980 | 20 |
| change-frequency-weak | 128 | 1.18 @ 272 | not_confirmed | — | — | [61.8; 320] | 0 | 150 | 850 |
| change-frequency-medium | 128 | 1.44 @ 16 | recovered | 344 | 2.688 | [56; 440] | 0 | 470 | 530 |
| change-frequency-strong | 128 | 1.96 @ 16 | recovered | 88 | 0.688 | [64; 376] | 0 | 939 | 61 |
| change-sensor_bias-weak | 128 | 1.12 @ 168 | recovered | 184 | 1.438 | [64; 384] | 98 | 799 | 103 |
| change-sensor_bias-medium | 128 | 1.25 @ 0 | not_confirmed | — | — | [16; 411] | 0 | 26 | 974 |
| change-sensor_bias-strong | 128 | 1.74 @ 0 | not_confirmed | — | — | [112; 416] | 0 | 134 | 866 |
| drift-mean-weak ≈ | 128 | 1.29 @ 368 | recovered | 392 | 3.062 | [376; 440] | 0 | 671 | 329 |
| drift-mean-medium ≈ | 128 | 1.20 @ 64 | recovered | 88 | 0.688 | [72; 440] | 0 | 847 | 153 |
| drift-mean-strong ≈ | 128 | 1.25 @ 344 | recovered | 384 | 3.000 | [167.6; 440] | 0 | 835 | 165 |
| drift-frequency-weak ≈ | 128 | 1.17 @ 64 | not_confirmed | — | — | [122.6; 440] | 0 | 374 | 626 |
| drift-frequency-medium ≈ | 128 | 1.20 @ 440 | not_confirmed | — | — | [128; 384] | 0 | 185 | 815 |
| drift-frequency-strong ≈ | 128 | 1.69 @ 80 | not_confirmed | — | — | [128; 416] | 0 | 261 | 739 |
| anomaly-spike-weak | 128 | 1.22 @ 184 | recovered | 280 | 2.188 | [8; 432] | 13 | 463 | 524 |
| anomaly-spike-medium | 128 | 1.82 @ 0 | not_confirmed | — | — | [16; 440] | 0 | 292 | 708 |
| anomaly-spike-strong | 128 | 3.11 @ 0 | recovered | 352 | 2.750 | [200; 440] | 0 | 485 | 515 |
| anomaly-freeze-weak | 128 | 1.26 @ 24 | recovered | 376 | 2.938 | [32; 392] | 3 | 441 | 556 |
| anomaly-freeze-medium | 128 | 1.22 @ 104 | recovered | 312 | 2.438 | [112; 440] | 0 | 592 | 408 |
| anomaly-freeze-strong | 128 | 2.00 @ 136 | not_confirmed | — | — | [184; 440] | 0 | 121 | 879 |
| none | 128 | 1.12 @ 464 | not_confirmed | — | — | [8; 424] | 15 | 463 | 522 |
| change-mean-weak | 64 | 1.32 @ 232 | not_confirmed | — | — | [268.8; 432] | 0 | 25 | 975 |
| change-mean-medium | 64 | 1.22 @ 0 | not_confirmed | — | — | [304; 432] | 0 | 105 | 895 |
| change-mean-strong | 64 | 1.31 @ 0 | not_confirmed | — | — | [128; 432] | 0 | 176 | 824 |
| change-variance-weak | 64 | 1.22 @ 40 | not_confirmed | — | — | [305.8; 390.2] | 0 | 10 | 990 |
| change-variance-medium | 64 | 1.11 @ 416 | not_confirmed | — | — | [272; 440] | 0 | 93 | 907 |
| change-variance-strong | 64 | 1.21 @ 208 | not_confirmed | — | — | [136; 416] | 0 | 110 | 890 |
| change-trend-weak | 64 | 1.28 @ 376 | not_confirmed | — | — | [400; 400] | 0 | 1 | 999 |
| change-trend-medium | 64 | 1.22 @ 56 | not_confirmed | — | — | [333.2; 440] | 0 | 174 | 826 |
| change-trend-strong | 64 | 1.20 @ 216 | not_confirmed | — | — | [336; 424] | 0 | 336 | 664 |
| change-ar-weak | 64 | 1.25 @ 0 | not_confirmed | — | — | [248; 263.2] | 0 | 3 | 997 |
| change-ar-medium | 64 | 1.18 @ 0 | not_confirmed | — | — | [238; 440] | 0 | 36 | 964 |
| change-ar-strong | 64 | 1.21 @ 16 | not_confirmed | — | — | [181.2; 427.6] | 0 | 52 | 948 |
| change-seasonality-weak | 64 | 1.34 @ 16 | not_confirmed | — | — | [320; 424] | 0 | 207 | 793 |
| change-seasonality-medium | 64 | 1.37 @ 0 | not_confirmed | — | — | [102.4; 422.4] | 0 | 105 | 895 |
| change-seasonality-strong | 64 | 1.58 @ 0 | recovered | 240 | 3.750 | [32; 352] | 0 | 579 | 421 |
| change-frequency-weak | 64 | 1.10 @ 136 | not_confirmed | — | — | [157.2; 320] | 0 | 27 | 973 |
| change-frequency-medium | 64 | 1.33 @ 24 | not_confirmed | — | — | — | 0 | 0 | 1000 |
| change-frequency-strong | 64 | 1.61 @ 16 | recovered | 384 | 6.000 | [48; 416] | 0 | 502 | 498 |
| change-sensor_bias-weak | 64 | 1.22 @ 64 | not_confirmed | — | — | [98.4; 422.8] | 0 | 207 | 793 |
| change-sensor_bias-medium | 64 | 1.20 @ 504 | not_confirmed | — | — | — | 0 | 0 | 1000 |
| change-sensor_bias-strong | 64 | 1.26 @ 40 | not_confirmed | — | — | [384; 440] | 0 | 131 | 869 |
| drift-mean-weak ≈ | 64 | 1.27 @ 368 | not_confirmed | — | — | [384; 440] | 0 | 334 | 666 |
| drift-mean-medium ≈ | 64 | 1.24 @ 64 | recovered | 184 | 2.875 | [96; 440] | 0 | 357 | 643 |
| drift-mean-strong ≈ | 64 | 1.19 @ 344 | recovered | 384 | 6.000 | [352; 440] | 0 | 791 | 209 |
| drift-frequency-weak ≈ | 64 | 1.22 @ 144 | not_confirmed | — | — | [181.2; 280] | 0 | 3 | 997 |
| drift-frequency-medium ≈ | 64 | 1.16 @ 256 | not_confirmed | — | — | [8; 432] | 1 | 75 | 924 |
| drift-frequency-strong ≈ | 64 | 1.33 @ 72 | not_confirmed | — | — | [176; 433] | 0 | 76 | 924 |
| anomaly-spike-weak | 64 | 1.24 @ 0 | not_confirmed | — | — | [8; 440] | 0 | 229 | 771 |
| anomaly-spike-medium | 64 | 1.73 @ 0 | not_confirmed | — | — | [136; 440] | 0 | 135 | 865 |
| anomaly-spike-strong | 64 | 2.67 @ 0 | not_confirmed | — | — | [211.2; 432] | 0 | 47 | 953 |
| anomaly-freeze-weak | 64 | 1.21 @ 24 | not_confirmed | — | — | [221; 432] | 0 | 14 | 986 |
| anomaly-freeze-medium | 64 | 1.56 @ 56 | not_confirmed | — | — | [304; 440] | 0 | 218 | 782 |
| anomaly-freeze-strong | 64 | 1.88 @ 144 | not_confirmed | — | — | [393.2; 416] | 0 | 7 | 993 |
| none | 64 | 1.19 @ 112 | not_confirmed | — | — | [240; 431.2] | 0 | 125 | 875 |

## H1: H=64, Q95 (основной)

CI в таблице — только по recovered-повторам. Drift помечен `≈`.

| Ячейка | L | Сглаженный пик @ d | Статус | d* | d*/L | Условный CI95 d* | boot: нет дегр. | boot: recovered | boot: не подтв. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| change-mean-weak | 512 | 1.14 @ 64 | not_confirmed | — | — | [132.4; 419.6] | 0 | 23 | 977 |
| change-mean-medium | 512 | 1.30 @ 0 | not_confirmed | — | — | [264; 440] | 0 | 176 | 824 |
| change-mean-strong | 512 | 2.18 @ 0 | not_confirmed | — | — | [368; 440] | 0 | 213 | 787 |
| change-variance-weak | 512 | 1.07 @ 72 | not_confirmed | — | — | [64; 440] | 73 | 370 | 557 |
| change-variance-medium | 512 | 1.03 @ 248 | recovered | 264 | 0.516 | [56; 440] | 25 | 919 | 56 |
| change-variance-strong | 512 | 1.07 @ 56 | recovered | 112 | 0.219 | [72; 312] | 0 | 444 | 556 |
| change-trend-weak | 512 | 1.36 @ 168 | not_confirmed | — | — | [256; 384.2] | 0 | 58 | 942 |
| change-trend-medium | 512 | 2.03 @ 120 | recovered | 336 | 0.656 | [312; 426] | 0 | 456 | 544 |
| change-trend-strong | 512 | 2.93 @ 80 | not_confirmed | — | — | [368; 432] | 0 | 148 | 852 |
| change-ar-weak | 512 | 1.10 @ 176 | not_confirmed | — | — | [208; 331] | 0 | 366 | 634 |
| change-ar-medium | 512 | 1.09 @ 272 | recovered | 440 | 0.859 | [328; 440] | 0 | 588 | 412 |
| change-ar-strong | 512 | 1.06 @ 16 | not_confirmed | — | — | [48; 432] | 0 | 170 | 830 |
| change-seasonality-weak | 512 | 1.29 @ 0 | recovered | 368 | 0.719 | [32; 432] | 0 | 446 | 554 |
| change-seasonality-medium | 512 | 1.37 @ 24 | recovered | 328 | 0.641 | [114.4; 382.8] | 0 | 567 | 433 |
| change-seasonality-strong | 512 | 1.59 @ 0 | recovered | 192 | 0.375 | [80; 408] | 0 | 918 | 82 |
| change-frequency-weak | 512 | 1.56 @ 104 | not_confirmed | — | — | [280; 440] | 0 | 362 | 638 |
| change-frequency-medium | 512 | 2.21 @ 40 | recovered | 440 | 0.859 | [280; 440] | 0 | 639 | 361 |
| change-frequency-strong | 512 | 2.48 @ 16 | recovered | 408 | 0.797 | [312; 440] | 0 | 407 | 593 |
| change-sensor_bias-weak | 512 | 1.15 @ 40 | not_confirmed | — | — | [96; 368] | 0 | 154 | 846 |
| change-sensor_bias-medium | 512 | 1.46 @ 0 | not_confirmed | — | — | [176; 440] | 0 | 34 | 966 |
| change-sensor_bias-strong | 512 | 2.30 @ 0 | not_confirmed | — | — | [372.6; 435.4] | 0 | 64 | 936 |
| drift-mean-weak ≈ | 512 | 1.19 @ 320 | recovered | 432 | 0.844 | [344; 440] | 0 | 571 | 429 |
| drift-mean-medium ≈ | 512 | 1.32 @ 48 | recovered | 264 | 0.516 | [176; 440] | 0 | 633 | 367 |
| drift-mean-strong ≈ | 512 | 1.85 @ 88 | not_confirmed | — | — | [288; 384] | 0 | 331 | 669 |
| drift-frequency-weak ≈ | 512 | 1.66 @ 136 | recovered | 432 | 0.844 | [336; 440] | 0 | 340 | 660 |
| drift-frequency-medium ≈ | 512 | 2.01 @ 112 | not_confirmed | — | — | [432; 440] | 0 | 6 | 994 |
| drift-frequency-strong ≈ | 512 | 2.08 @ 120 | not_confirmed | — | — | [304; 350.6] | 0 | 8 | 992 |
| anomaly-spike-weak | 512 | 1.12 @ 224 | recovered | 264 | 0.516 | [16; 424] | 50 | 586 | 364 |
| anomaly-spike-medium | 512 | 1.40 @ 0 | not_confirmed | — | — | [16; 440] | 0 | 435 | 565 |
| anomaly-spike-strong | 512 | 2.36 @ 0 | not_confirmed | — | — | [158.4; 440] | 0 | 353 | 647 |
| anomaly-freeze-weak | 512 | 1.08 @ 8 | not_confirmed | — | — | [16; 432] | 14 | 261 | 725 |
| anomaly-freeze-medium | 512 | 1.67 @ 40 | not_confirmed | — | — | [200; 440] | 0 | 317 | 683 |
| anomaly-freeze-strong | 512 | 1.61 @ 112 | not_confirmed | — | — | [144; 432] | 0 | 222 | 778 |
| none | 512 | 1.07 @ 80 | not_confirmed | — | — | [43.4; 396.6] | 99 | 298 | 603 |
| change-mean-weak | 256 | 1.14 @ 504 | not_confirmed | — | — | [32; 272] | 100 | 272 | 628 |
| change-mean-medium | 256 | 1.28 @ 0 | recovered | 96 | 0.375 | [16; 408] | 0 | 919 | 81 |
| change-mean-strong | 256 | 1.87 @ 0 | recovered | 240 | 0.938 | [168; 352] | 0 | 861 | 139 |
| change-variance-weak | 256 | 1.10 @ 240 | recovered | 248 | 0.969 | [64; 440] | 292 | 573 | 135 |
| change-variance-medium | 256 | 1.07 @ 248 | no_degradation | — | — | [40; 440] | 313 | 679 | 8 |
| change-variance-strong | 256 | 1.03 @ 200 | no_degradation | — | — | [64; 408] | 446 | 478 | 76 |
| change-trend-weak | 256 | 1.27 @ 448 | not_confirmed | — | — | [226.2; 336] | 0 | 12 | 988 |
| change-trend-medium | 256 | 1.41 @ 264 | not_confirmed | — | — | [217; 404.2] | 0 | 100 | 900 |
| change-trend-strong | 256 | 1.89 @ 72 | recovered | 376 | 1.469 | [232; 408] | 0 | 905 | 95 |
| change-ar-weak | 256 | 1.11 @ 16 | recovered | 24 | 0.094 | [24; 257.4] | 176 | 634 | 190 |
| change-ar-medium | 256 | 1.12 @ 72 | recovered | 112 | 0.438 | [80; 432] | 84 | 903 | 13 |
| change-ar-strong | 256 | 1.07 @ 40 | no_degradation | — | — | [16; 424] | 435 | 463 | 102 |
| change-seasonality-weak | 256 | 1.36 @ 8 | recovered | 168 | 0.656 | [32; 440] | 0 | 824 | 176 |
| change-seasonality-medium | 256 | 1.39 @ 0 | recovered | 192 | 0.750 | [48; 312] | 0 | 783 | 217 |
| change-seasonality-strong | 256 | 1.49 @ 0 | recovered | 168 | 0.656 | [32; 192] | 0 | 1000 | 0 |
| change-frequency-weak | 256 | 1.29 @ 96 | recovered | 168 | 0.656 | [144; 416] | 0 | 772 | 228 |
| change-frequency-medium | 256 | 2.01 @ 32 | recovered | 400 | 1.562 | [144; 440] | 0 | 735 | 265 |
| change-frequency-strong | 256 | 2.15 @ 8 | recovered | 192 | 0.750 | [160; 224] | 0 | 697 | 303 |
| change-sensor_bias-weak | 256 | 1.16 @ 168 | recovered | 192 | 0.750 | [64; 384] | 1 | 446 | 553 |
| change-sensor_bias-medium | 256 | 1.33 @ 0 | recovered | 184 | 0.719 | [24; 424] | 0 | 417 | 583 |
| change-sensor_bias-strong | 256 | 1.87 @ 0 | recovered | 376 | 1.469 | [248; 405] | 0 | 536 | 464 |
| drift-mean-weak ≈ | 256 | 1.17 @ 328 | recovered | 368 | 1.438 | [40; 440] | 14 | 926 | 60 |
| drift-mean-medium ≈ | 256 | 1.29 @ 48 | recovered | 264 | 1.031 | [72; 440] | 0 | 959 | 41 |
| drift-mean-strong ≈ | 256 | 1.52 @ 80 | recovered | 336 | 1.312 | [259; 376] | 0 | 896 | 104 |
| drift-frequency-weak ≈ | 256 | 1.38 @ 136 | recovered | 296 | 1.156 | [168; 432] | 0 | 533 | 467 |
| drift-frequency-medium ≈ | 256 | 1.76 @ 112 | not_confirmed | — | — | [232; 440] | 0 | 209 | 791 |
| drift-frequency-strong ≈ | 256 | 1.81 @ 48 | recovered | 304 | 1.188 | [256; 432] | 0 | 764 | 236 |
| anomaly-spike-weak | 256 | 1.04 @ 224 | no_degradation | — | — | [16; 288] | 602 | 377 | 21 |
| anomaly-spike-medium | 256 | 1.39 @ 0 | recovered | 16 | 0.062 | [8; 427] | 6 | 706 | 288 |
| anomaly-spike-strong | 256 | 2.21 @ 0 | recovered | 392 | 1.531 | [16; 440] | 0 | 842 | 158 |
| anomaly-freeze-weak | 256 | 1.11 @ 8 | recovered | 16 | 0.062 | [16; 408] | 247 | 530 | 223 |
| anomaly-freeze-medium | 256 | 1.56 @ 40 | recovered | 424 | 1.656 | [64; 440] | 0 | 738 | 262 |
| anomaly-freeze-strong | 256 | 1.64 @ 120 | recovered | 400 | 1.562 | [144; 432] | 0 | 441 | 559 |
| none | 256 | 1.07 @ 112 | no_degradation | — | — | [59; 352] | 418 | 364 | 218 |
| change-mean-weak | 128 | 1.15 @ 504 | not_confirmed | — | — | [120; 424] | 1 | 109 | 890 |
| change-mean-medium | 128 | 1.21 @ 504 | not_confirmed | — | — | [8; 432] | 0 | 107 | 893 |
| change-mean-strong | 128 | 1.54 @ 0 | not_confirmed | — | — | [80; 440] | 0 | 446 | 554 |
| change-variance-weak | 128 | 1.10 @ 192 | not_confirmed | — | — | [64; 432] | 49 | 342 | 609 |
| change-variance-medium | 128 | 1.20 @ 264 | recovered | 304 | 2.375 | [288; 432] | 0 | 776 | 224 |
| change-variance-strong | 128 | 1.07 @ 328 | recovered | 336 | 2.625 | [64; 408] | 21 | 894 | 85 |
| change-trend-weak | 128 | 1.13 @ 448 | not_confirmed | — | — | [80; 300] | 0 | 72 | 928 |
| change-trend-medium | 128 | 1.21 @ 264 | recovered | 320 | 2.500 | [312; 440] | 0 | 788 | 212 |
| change-trend-strong | 128 | 1.38 @ 56 | recovered | 408 | 3.188 | [376; 440] | 0 | 511 | 489 |
| change-ar-weak | 128 | 1.18 @ 0 | not_confirmed | — | — | [15.2; 352] | 1 | 157 | 842 |
| change-ar-medium | 128 | 1.19 @ 16 | not_confirmed | — | — | [280; 440] | 0 | 101 | 899 |
| change-ar-strong | 128 | 1.21 @ 0 | recovered | 408 | 3.188 | [240; 440] | 0 | 307 | 693 |
| change-seasonality-weak | 128 | 1.34 @ 16 | recovered | 48 | 0.375 | [40; 424] | 0 | 683 | 317 |
| change-seasonality-medium | 128 | 1.36 @ 0 | recovered | 272 | 2.125 | [92.4; 416] | 0 | 583 | 417 |
| change-seasonality-strong | 128 | 1.43 @ 0 | recovered | 64 | 0.500 | [24; 248] | 0 | 976 | 24 |
| change-frequency-weak | 128 | 1.15 @ 152 | not_confirmed | — | — | [145.6; 304] | 3 | 285 | 712 |
| change-frequency-medium | 128 | 1.69 @ 16 | recovered | 416 | 3.250 | [64; 416] | 0 | 471 | 529 |
| change-frequency-strong | 128 | 1.88 @ 16 | recovered | 104 | 0.812 | [88; 336] | 0 | 845 | 155 |
| change-sensor_bias-weak | 128 | 1.12 @ 176 | not_confirmed | — | — | [72; 376] | 2 | 301 | 697 |
| change-sensor_bias-medium | 128 | 1.27 @ 0 | not_confirmed | — | — | [272; 416] | 0 | 20 | 980 |
| change-sensor_bias-strong | 128 | 1.51 @ 0 | not_confirmed | — | — | [112; 432] | 0 | 138 | 862 |
| drift-mean-weak ≈ | 128 | 1.21 @ 336 | recovered | 376 | 2.938 | [360; 424] | 0 | 893 | 107 |
| drift-mean-medium ≈ | 128 | 1.18 @ 48 | recovered | 232 | 1.812 | [48; 440] | 0 | 906 | 94 |
| drift-mean-strong ≈ | 128 | 1.38 @ 96 | recovered | 400 | 3.125 | [288; 432] | 0 | 514 | 486 |
| drift-frequency-weak ≈ | 128 | 1.22 @ 112 | not_confirmed | — | — | [181.6; 328] | 0 | 7 | 993 |
| drift-frequency-medium ≈ | 128 | 1.41 @ 104 | not_confirmed | — | — | [152; 436.6] | 0 | 18 | 982 |
| drift-frequency-strong ≈ | 128 | 1.78 @ 40 | not_confirmed | — | — | [172.8; 432] | 0 | 185 | 815 |
| anomaly-spike-weak | 128 | 1.02 @ 0 | no_degradation | — | — | [8; 304] | 325 | 596 | 79 |
| anomaly-spike-medium | 128 | 1.43 @ 0 | recovered | 16 | 0.125 | [8; 416] | 12 | 502 | 486 |
| anomaly-spike-strong | 128 | 2.26 @ 0 | not_confirmed | — | — | [139.2; 440] | 0 | 57 | 943 |
| anomaly-freeze-weak | 128 | 1.21 @ 8 | recovered | 272 | 2.125 | [24; 392] | 3 | 634 | 363 |
| anomaly-freeze-medium | 128 | 1.41 @ 40 | recovered | 416 | 3.250 | [163.8; 432] | 0 | 700 | 300 |
| anomaly-freeze-strong | 128 | 1.96 @ 136 | not_confirmed | — | — | [208; 432] | 0 | 157 | 843 |
| none | 128 | 1.02 @ 256 | no_degradation | — | — | [24; 440] | 224 | 545 | 231 |
| change-mean-weak | 64 | 1.16 @ 232 | not_confirmed | — | — | [186.2; 410.8] | 5 | 172 | 823 |
| change-mean-medium | 64 | 1.25 @ 504 | not_confirmed | — | — | [384; 384] | 0 | 1 | 999 |
| change-mean-strong | 64 | 1.30 @ 8 | not_confirmed | — | — | [26; 432] | 0 | 306 | 694 |
| change-variance-weak | 64 | 1.13 @ 192 | not_confirmed | — | — | [64; 432] | 21 | 349 | 630 |
| change-variance-medium | 64 | 1.12 @ 264 | recovered | 296 | 4.625 | [280; 440] | 3 | 763 | 234 |
| change-variance-strong | 64 | 1.15 @ 216 | recovered | 232 | 3.625 | [104; 440] | 0 | 657 | 343 |
| change-trend-weak | 64 | 1.18 @ 208 | not_confirmed | — | — | [190; 371] | 0 | 26 | 974 |
| change-trend-medium | 64 | 1.16 @ 320 | recovered | 336 | 5.250 | [328; 400] | 0 | 757 | 243 |
| change-trend-strong | 64 | 1.15 @ 368 | recovered | 392 | 6.125 | [368; 432] | 0 | 449 | 551 |
| change-ar-weak | 64 | 1.26 @ 0 | not_confirmed | — | — | [247; 353] | 0 | 36 | 964 |
| change-ar-medium | 64 | 1.13 @ 0 | not_confirmed | — | — | [81; 440] | 0 | 86 | 914 |
| change-ar-strong | 64 | 1.16 @ 8 | not_confirmed | — | — | [32; 440] | 0 | 237 | 763 |
| change-seasonality-weak | 64 | 1.33 @ 16 | not_confirmed | — | — | [72; 408] | 0 | 298 | 702 |
| change-seasonality-medium | 64 | 1.35 @ 0 | not_confirmed | — | — | [64; 402.2] | 0 | 270 | 730 |
| change-seasonality-strong | 64 | 1.33 @ 0 | recovered | 48 | 0.750 | [24; 248] | 0 | 675 | 325 |
| change-frequency-weak | 64 | 1.06 @ 456 | not_confirmed | — | — | [8; 408] | 38 | 315 | 647 |
| change-frequency-medium | 64 | 1.43 @ 0 | not_confirmed | — | — | [336.6; 352] | 0 | 4 | 996 |
| change-frequency-strong | 64 | 1.61 @ 8 | recovered | 80 | 1.250 | [40; 392.8] | 0 | 479 | 521 |
| change-sensor_bias-weak | 64 | 1.05 @ 64 | recovered | 72 | 1.125 | [72; 352] | 44 | 659 | 297 |
| change-sensor_bias-medium | 64 | 1.21 @ 464 | not_confirmed | — | — | [376; 376] | 0 | 1 | 999 |
| change-sensor_bias-strong | 64 | 1.35 @ 0 | not_confirmed | — | — | [144; 436.4] | 0 | 99 | 901 |
| drift-mean-weak ≈ | 64 | 1.22 @ 336 | recovered | 392 | 6.125 | [376; 432] | 0 | 534 | 466 |
| drift-mean-medium ≈ | 64 | 1.16 @ 48 | recovered | 440 | 6.875 | [64; 440] | 0 | 608 | 392 |
| drift-mean-strong ≈ | 64 | 1.29 @ 24 | recovered | 384 | 6.000 | [344; 400] | 0 | 803 | 197 |
| drift-frequency-weak ≈ | 64 | 1.19 @ 128 | not_confirmed | — | — | [216.4; 384] | 0 | 44 | 956 |
| drift-frequency-medium ≈ | 64 | 1.11 @ 472 | not_confirmed | — | — | [80; 432.2] | 3 | 160 | 837 |
| drift-frequency-strong ≈ | 64 | 1.65 @ 80 | not_confirmed | — | — | [168.4; 352] | 0 | 7 | 993 |
| anomaly-spike-weak | 64 | 1.10 @ 272 | recovered | 296 | 4.625 | [8; 424] | 107 | 496 | 397 |
| anomaly-spike-medium | 64 | 1.45 @ 0 | not_confirmed | — | — | [16; 432] | 2 | 340 | 658 |
| anomaly-spike-strong | 64 | 1.92 @ 0 | not_confirmed | — | — | [294; 424] | 0 | 31 | 969 |
| anomaly-freeze-weak | 64 | 1.16 @ 56 | recovered | 192 | 3.000 | [64; 352] | 0 | 430 | 570 |
| anomaly-freeze-medium | 64 | 1.80 @ 56 | not_confirmed | — | — | [224; 432] | 0 | 246 | 754 |
| anomaly-freeze-strong | 64 | 1.84 @ 144 | not_confirmed | — | — | [208; 208] | 0 | 1 | 999 |
| none | 64 | 1.09 @ 104 | recovered | 120 | 1.875 | [87.6; 440] | 69 | 307 | 624 |

## H1: H=64, Q90 (чувствительность)

CI в таблице — только по recovered-повторам. Drift помечен `≈`.

| Ячейка | L | Сглаженный пик @ d | Статус | d* | d*/L | Условный CI95 d* | boot: нет дегр. | boot: recovered | boot: не подтв. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| change-mean-weak | 512 | 1.14 @ 64 | not_confirmed | — | — | [144; 421.6] | 0 | 13 | 987 |
| change-mean-medium | 512 | 1.30 @ 0 | not_confirmed | — | — | [272; 440] | 0 | 107 | 893 |
| change-mean-strong | 512 | 2.18 @ 0 | not_confirmed | — | — | [376; 440] | 0 | 168 | 832 |
| change-variance-weak | 512 | 1.07 @ 72 | not_confirmed | — | — | [64; 440] | 48 | 302 | 650 |
| change-variance-medium | 512 | 1.03 @ 248 | recovered | 264 | 0.516 | [120; 440] | 4 | 910 | 86 |
| change-variance-strong | 512 | 1.07 @ 56 | not_confirmed | — | — | [80; 315.6] | 0 | 343 | 657 |
| change-trend-weak | 512 | 1.36 @ 168 | not_confirmed | — | — | [336; 440] | 0 | 25 | 975 |
| change-trend-medium | 512 | 2.03 @ 120 | not_confirmed | — | — | [315.6; 432] | 0 | 379 | 621 |
| change-trend-strong | 512 | 2.93 @ 80 | not_confirmed | — | — | [376; 432] | 0 | 110 | 890 |
| change-ar-weak | 512 | 1.10 @ 176 | not_confirmed | — | — | [216; 339.4] | 0 | 264 | 736 |
| change-ar-medium | 512 | 1.09 @ 272 | recovered | 440 | 0.859 | [344; 440] | 0 | 434 | 566 |
| change-ar-strong | 512 | 1.06 @ 16 | not_confirmed | — | — | [55.6; 416.8] | 0 | 79 | 921 |
| change-seasonality-weak | 512 | 1.29 @ 0 | not_confirmed | — | — | [47.8; 440] | 0 | 360 | 640 |
| change-seasonality-medium | 512 | 1.37 @ 24 | not_confirmed | — | — | [128; 384] | 0 | 444 | 556 |
| change-seasonality-strong | 512 | 1.59 @ 0 | recovered | 200 | 0.391 | [85.6; 408] | 0 | 869 | 131 |
| change-frequency-weak | 512 | 1.56 @ 104 | not_confirmed | — | — | [280; 440] | 0 | 295 | 705 |
| change-frequency-medium | 512 | 2.21 @ 40 | recovered | 440 | 0.859 | [280; 440] | 0 | 561 | 439 |
| change-frequency-strong | 512 | 2.48 @ 16 | not_confirmed | — | — | [336; 440] | 0 | 316 | 684 |
| change-sensor_bias-weak | 512 | 1.15 @ 40 | not_confirmed | — | — | [128; 368] | 0 | 111 | 889 |
| change-sensor_bias-medium | 512 | 1.46 @ 0 | not_confirmed | — | — | [179.8; 440] | 0 | 20 | 980 |
| change-sensor_bias-strong | 512 | 2.30 @ 0 | not_confirmed | — | — | [376; 440] | 0 | 33 | 967 |
| drift-mean-weak ≈ | 512 | 1.19 @ 320 | recovered | 432 | 0.844 | [344; 440] | 0 | 470 | 530 |
| drift-mean-medium ≈ | 512 | 1.32 @ 48 | recovered | 336 | 0.656 | [184; 440] | 0 | 538 | 462 |
| drift-mean-strong ≈ | 512 | 1.85 @ 88 | not_confirmed | — | — | [288; 384] | 0 | 243 | 757 |
| drift-frequency-weak ≈ | 512 | 1.66 @ 136 | recovered | 440 | 0.859 | [344; 440] | 0 | 235 | 765 |
| drift-frequency-medium ≈ | 512 | 2.01 @ 112 | not_confirmed | — | — | [432.4; 440] | 0 | 3 | 997 |
| drift-frequency-strong ≈ | 512 | 2.08 @ 120 | not_confirmed | — | — | [304.8; 349.6] | 0 | 5 | 995 |
| anomaly-spike-weak | 512 | 1.12 @ 224 | recovered | 264 | 0.516 | [16; 432] | 32 | 535 | 433 |
| anomaly-spike-medium | 512 | 1.40 @ 0 | not_confirmed | — | — | [16; 432] | 0 | 377 | 623 |
| anomaly-spike-strong | 512 | 2.36 @ 0 | not_confirmed | — | — | [336; 440] | 0 | 282 | 718 |
| anomaly-freeze-weak | 512 | 1.08 @ 8 | not_confirmed | — | — | [16; 440] | 6 | 210 | 784 |
| anomaly-freeze-medium | 512 | 1.67 @ 40 | not_confirmed | — | — | [208; 440] | 0 | 251 | 749 |
| anomaly-freeze-strong | 512 | 1.61 @ 112 | not_confirmed | — | — | [144; 432] | 0 | 157 | 843 |
| none | 512 | 1.07 @ 80 | not_confirmed | — | — | [40; 392] | 62 | 286 | 652 |
| change-mean-weak | 256 | 1.14 @ 504 | not_confirmed | — | — | [46.6; 273.4] | 8 | 154 | 838 |
| change-mean-medium | 256 | 1.28 @ 0 | recovered | 288 | 1.125 | [32; 432] | 0 | 681 | 319 |
| change-mean-strong | 256 | 1.87 @ 0 | recovered | 344 | 1.344 | [230; 416] | 0 | 631 | 369 |
| change-variance-weak | 256 | 1.10 @ 240 | recovered | 424 | 1.656 | [72; 440] | 77 | 511 | 412 |
| change-variance-medium | 256 | 1.07 @ 248 | recovered | 288 | 1.125 | [48; 440] | 74 | 868 | 58 |
| change-variance-strong | 256 | 1.03 @ 200 | no_degradation | — | — | [64; 408] | 73 | 711 | 216 |
| change-trend-weak | 256 | 1.27 @ 448 | not_confirmed | — | — | — | 0 | 0 | 1000 |
| change-trend-medium | 256 | 1.41 @ 264 | not_confirmed | — | — | [312; 408] | 0 | 28 | 972 |
| change-trend-strong | 256 | 1.89 @ 72 | recovered | 376 | 1.469 | [253; 416] | 0 | 786 | 214 |
| change-ar-weak | 256 | 1.11 @ 16 | recovered | 232 | 0.906 | [103; 352] | 1 | 516 | 483 |
| change-ar-medium | 256 | 1.12 @ 72 | recovered | 296 | 1.156 | [112; 440] | 0 | 872 | 128 |
| change-ar-strong | 256 | 1.07 @ 40 | recovered | 80 | 0.312 | [32; 424] | 10 | 478 | 512 |
| change-seasonality-weak | 256 | 1.36 @ 8 | recovered | 368 | 1.438 | [152; 440] | 0 | 590 | 410 |
| change-seasonality-medium | 256 | 1.39 @ 0 | not_confirmed | — | — | [88; 352] | 0 | 465 | 535 |
| change-seasonality-strong | 256 | 1.49 @ 0 | recovered | 176 | 0.688 | [40; 208] | 0 | 993 | 7 |
| change-frequency-weak | 256 | 1.29 @ 96 | recovered | 176 | 0.688 | [152; 424] | 0 | 527 | 473 |
| change-frequency-medium | 256 | 2.01 @ 32 | not_confirmed | — | — | [168; 440] | 0 | 350 | 650 |
| change-frequency-strong | 256 | 2.15 @ 8 | recovered | 200 | 0.781 | [168; 352] | 0 | 384 | 616 |
| change-sensor_bias-weak | 256 | 1.16 @ 168 | not_confirmed | — | — | [184; 400] | 0 | 159 | 841 |
| change-sensor_bias-medium | 256 | 1.33 @ 0 | not_confirmed | — | — | [184; 432] | 0 | 138 | 862 |
| change-sensor_bias-strong | 256 | 1.87 @ 0 | not_confirmed | — | — | [352; 408] | 0 | 285 | 715 |
| drift-mean-weak ≈ | 256 | 1.17 @ 328 | recovered | 376 | 1.469 | [143.4; 440] | 1 | 758 | 241 |
| drift-mean-medium ≈ | 256 | 1.29 @ 48 | recovered | 424 | 1.656 | [184; 440] | 0 | 814 | 186 |
| drift-mean-strong ≈ | 256 | 1.52 @ 80 | recovered | 352 | 1.375 | [288; 392] | 0 | 604 | 396 |
| drift-frequency-weak ≈ | 256 | 1.38 @ 136 | not_confirmed | — | — | [240; 432] | 0 | 128 | 872 |
| drift-frequency-medium ≈ | 256 | 1.76 @ 112 | not_confirmed | — | — | [315.2; 440] | 0 | 23 | 977 |
| drift-frequency-strong ≈ | 256 | 1.81 @ 48 | not_confirmed | — | — | [272; 432] | 0 | 453 | 547 |
| anomaly-spike-weak | 256 | 1.04 @ 224 | no_degradation | — | — | [16; 296] | 410 | 519 | 71 |
| anomaly-spike-medium | 256 | 1.39 @ 0 | recovered | 416 | 1.625 | [16; 432] | 0 | 522 | 478 |
| anomaly-spike-strong | 256 | 2.21 @ 0 | recovered | 416 | 1.625 | [72; 440] | 0 | 641 | 359 |
| anomaly-freeze-weak | 256 | 1.11 @ 8 | recovered | 32 | 0.125 | [16; 416] | 67 | 556 | 377 |
| anomaly-freeze-medium | 256 | 1.56 @ 40 | recovered | 432 | 1.688 | [131; 440] | 0 | 500 | 500 |
| anomaly-freeze-strong | 256 | 1.64 @ 120 | not_confirmed | — | — | [176; 432] | 0 | 151 | 849 |
| none | 256 | 1.07 @ 112 | recovered | 128 | 0.500 | [56; 360] | 239 | 383 | 378 |
| change-mean-weak | 128 | 1.15 @ 504 | not_confirmed | — | — | [176; 424] | 0 | 41 | 959 |
| change-mean-medium | 128 | 1.21 @ 504 | not_confirmed | — | — | [15; 296] | 0 | 36 | 964 |
| change-mean-strong | 128 | 1.54 @ 0 | not_confirmed | — | — | [88; 440] | 0 | 281 | 719 |
| change-variance-weak | 128 | 1.10 @ 192 | not_confirmed | — | — | [64; 432] | 15 | 211 | 774 |
| change-variance-medium | 128 | 1.20 @ 264 | recovered | 424 | 3.312 | [304; 440] | 0 | 483 | 517 |
| change-variance-strong | 128 | 1.07 @ 328 | recovered | 336 | 2.625 | [72; 440] | 0 | 772 | 228 |
| change-trend-weak | 128 | 1.13 @ 448 | not_confirmed | — | — | [214.4; 369.6] | 0 | 17 | 983 |
| change-trend-medium | 128 | 1.21 @ 264 | recovered | 360 | 2.812 | [312; 440] | 0 | 623 | 377 |
| change-trend-strong | 128 | 1.38 @ 56 | recovered | 416 | 3.250 | [376; 440] | 0 | 378 | 622 |
| change-ar-weak | 128 | 1.18 @ 0 | not_confirmed | — | — | [205; 360] | 0 | 76 | 924 |
| change-ar-medium | 128 | 1.19 @ 16 | not_confirmed | — | — | [352; 440] | 0 | 15 | 985 |
| change-ar-strong | 128 | 1.21 @ 0 | not_confirmed | — | — | [374.4; 440] | 0 | 77 | 923 |
| change-seasonality-weak | 128 | 1.34 @ 16 | recovered | 128 | 1.000 | [40; 424] | 0 | 519 | 481 |
| change-seasonality-medium | 128 | 1.36 @ 0 | not_confirmed | — | — | [120; 419] | 0 | 386 | 614 |
| change-seasonality-strong | 128 | 1.43 @ 0 | recovered | 72 | 0.562 | [24; 248] | 0 | 939 | 61 |
| change-frequency-weak | 128 | 1.15 @ 152 | not_confirmed | — | — | [160; 322] | 2 | 159 | 839 |
| change-frequency-medium | 128 | 1.69 @ 16 | not_confirmed | — | — | [67.2; 424] | 0 | 337 | 663 |
| change-frequency-strong | 128 | 1.88 @ 16 | recovered | 104 | 0.812 | [88; 376] | 0 | 672 | 328 |
| change-sensor_bias-weak | 128 | 1.12 @ 176 | not_confirmed | — | — | [80; 384] | 0 | 108 | 892 |
| change-sensor_bias-medium | 128 | 1.27 @ 0 | not_confirmed | — | — | [392.2; 399.8] | 0 | 2 | 998 |
| change-sensor_bias-strong | 128 | 1.51 @ 0 | not_confirmed | — | — | [244.8; 403.8] | 0 | 22 | 978 |
| drift-mean-weak ≈ | 128 | 1.21 @ 336 | recovered | 384 | 3.000 | [360; 432] | 0 | 733 | 267 |
| drift-mean-medium ≈ | 128 | 1.18 @ 48 | recovered | 232 | 1.812 | [56; 440] | 0 | 745 | 255 |
| drift-mean-strong ≈ | 128 | 1.38 @ 96 | not_confirmed | — | — | [288; 432] | 0 | 318 | 682 |
| drift-frequency-weak ≈ | 128 | 1.22 @ 112 | not_confirmed | — | — | [329; 367] | 0 | 2 | 998 |
| drift-frequency-medium ≈ | 128 | 1.41 @ 104 | not_confirmed | — | — | [160; 430.4] | 0 | 5 | 995 |
| drift-frequency-strong ≈ | 128 | 1.78 @ 40 | not_confirmed | — | — | [176; 432] | 0 | 81 | 919 |
| anomaly-spike-weak | 128 | 1.02 @ 0 | no_degradation | — | — | [8; 424] | 190 | 662 | 148 |
| anomaly-spike-medium | 128 | 1.43 @ 0 | not_confirmed | — | — | [8; 424] | 6 | 402 | 592 |
| anomaly-spike-strong | 128 | 2.26 @ 0 | not_confirmed | — | — | [219.6; 437.2] | 0 | 15 | 985 |
| anomaly-freeze-weak | 128 | 1.21 @ 8 | recovered | 288 | 2.250 | [25; 392] | 0 | 486 | 514 |
| anomaly-freeze-medium | 128 | 1.41 @ 40 | recovered | 424 | 3.312 | [216; 440] | 0 | 456 | 544 |
| anomaly-freeze-strong | 128 | 1.96 @ 136 | not_confirmed | — | — | [216; 432] | 0 | 37 | 963 |
| none | 128 | 1.02 @ 256 | recovered | 264 | 2.062 | [120; 440] | 90 | 569 | 341 |
| change-mean-weak | 64 | 1.16 @ 232 | not_confirmed | — | — | [192; 424] | 0 | 86 | 914 |
| change-mean-medium | 64 | 1.25 @ 504 | not_confirmed | — | — | — | 0 | 0 | 1000 |
| change-mean-strong | 64 | 1.30 @ 8 | not_confirmed | — | — | [109.6; 432] | 0 | 158 | 842 |
| change-variance-weak | 64 | 1.13 @ 192 | not_confirmed | — | — | [84.4; 424] | 2 | 183 | 815 |
| change-variance-medium | 64 | 1.12 @ 264 | recovered | 424 | 6.625 | [288; 440] | 0 | 527 | 473 |
| change-variance-strong | 64 | 1.15 @ 216 | recovered | 440 | 6.875 | [232; 440] | 0 | 405 | 595 |
| change-trend-weak | 64 | 1.18 @ 208 | not_confirmed | — | — | [232.6; 263.4] | 0 | 4 | 996 |
| change-trend-medium | 64 | 1.16 @ 320 | recovered | 336 | 5.250 | [328; 440] | 0 | 534 | 466 |
| change-trend-strong | 64 | 1.15 @ 368 | not_confirmed | — | — | [376; 432] | 0 | 278 | 722 |
| change-ar-weak | 64 | 1.26 @ 0 | not_confirmed | — | — | [250.4; 351.4] | 0 | 4 | 996 |
| change-ar-medium | 64 | 1.13 @ 0 | not_confirmed | — | — | [336; 350.8] | 0 | 4 | 996 |
| change-ar-strong | 64 | 1.16 @ 8 | not_confirmed | — | — | [245.8; 440] | 0 | 70 | 930 |
| change-seasonality-weak | 64 | 1.33 @ 16 | not_confirmed | — | — | [176; 416] | 0 | 169 | 831 |
| change-seasonality-medium | 64 | 1.35 @ 0 | not_confirmed | — | — | [112; 408] | 0 | 118 | 882 |
| change-seasonality-strong | 64 | 1.33 @ 0 | recovered | 48 | 0.750 | [24; 256] | 0 | 480 | 520 |
| change-frequency-weak | 64 | 1.06 @ 456 | not_confirmed | — | — | [8; 400] | 1 | 202 | 797 |
| change-frequency-medium | 64 | 1.43 @ 0 | not_confirmed | — | — | — | 0 | 0 | 1000 |
| change-frequency-strong | 64 | 1.61 @ 8 | not_confirmed | — | — | [40.6; 392] | 0 | 244 | 756 |
| change-sensor_bias-weak | 64 | 1.05 @ 64 | recovered | 80 | 1.250 | [72; 361] | 7 | 556 | 437 |
| change-sensor_bias-medium | 64 | 1.21 @ 464 | not_confirmed | — | — | — | 0 | 0 | 1000 |
| change-sensor_bias-strong | 64 | 1.35 @ 0 | not_confirmed | — | — | [216; 405] | 0 | 28 | 972 |
| drift-mean-weak ≈ | 64 | 1.22 @ 336 | not_confirmed | — | — | [376; 440] | 0 | 257 | 743 |
| drift-mean-medium ≈ | 64 | 1.16 @ 48 | recovered | 440 | 6.875 | [72; 440] | 0 | 377 | 623 |
| drift-mean-strong ≈ | 64 | 1.29 @ 24 | recovered | 392 | 6.125 | [352; 405.6] | 0 | 573 | 427 |
| drift-frequency-weak ≈ | 64 | 1.19 @ 128 | not_confirmed | — | — | [336; 384] | 0 | 11 | 989 |
| drift-frequency-medium ≈ | 64 | 1.11 @ 472 | not_confirmed | — | — | [128; 434.6] | 0 | 68 | 932 |
| drift-frequency-strong ≈ | 64 | 1.65 @ 80 | not_confirmed | — | — | [164.2; 359.4] | 0 | 4 | 996 |
| anomaly-spike-weak | 64 | 1.10 @ 272 | recovered | 296 | 4.625 | [8; 432] | 44 | 393 | 563 |
| anomaly-spike-medium | 64 | 1.45 @ 0 | not_confirmed | — | — | [16; 435] | 2 | 226 | 772 |
| anomaly-spike-strong | 64 | 1.92 @ 0 | not_confirmed | — | — | [373; 416] | 0 | 6 | 994 |
| anomaly-freeze-weak | 64 | 1.16 @ 56 | not_confirmed | — | — | [136; 362.4] | 0 | 269 | 731 |
| anomaly-freeze-medium | 64 | 1.80 @ 56 | not_confirmed | — | — | [304; 432] | 0 | 116 | 884 |
| anomaly-freeze-strong | 64 | 1.84 @ 144 | not_confirmed | — | — | — | 0 | 0 | 1000 |
| none | 64 | 1.09 @ 104 | not_confirmed | — | — | [96; 440] | 28 | 231 | 741 |

## Чувствительность: без drift

Состав основной оценки не меняется; таблица ниже исключает только приближённый эталон drift.

| L | H | Q | Статус | Ячеек |
| --- | --- | --- | --- | --- |
| 64 | 16 | 90 | not_confirmed | 26 |
| 64 | 16 | 90 | recovered | 2 |
| 64 | 16 | 95 | not_confirmed | 17 |
| 64 | 16 | 95 | recovered | 11 |
| 64 | 64 | 90 | not_confirmed | 22 |
| 64 | 64 | 90 | recovered | 6 |
| 64 | 64 | 95 | not_confirmed | 18 |
| 64 | 64 | 95 | recovered | 10 |
| 128 | 16 | 90 | not_confirmed | 10 |
| 128 | 16 | 90 | recovered | 18 |
| 128 | 16 | 95 | not_confirmed | 6 |
| 128 | 16 | 95 | recovered | 22 |
| 128 | 64 | 90 | no_degradation | 1 |
| 128 | 64 | 90 | not_confirmed | 17 |
| 128 | 64 | 90 | recovered | 10 |
| 128 | 64 | 95 | no_degradation | 2 |
| 128 | 64 | 95 | not_confirmed | 13 |
| 128 | 64 | 95 | recovered | 13 |
| 256 | 16 | 90 | not_confirmed | 15 |
| 256 | 16 | 90 | recovered | 13 |
| 256 | 16 | 95 | not_confirmed | 11 |
| 256 | 16 | 95 | recovered | 17 |
| 256 | 64 | 90 | no_degradation | 2 |
| 256 | 64 | 90 | not_confirmed | 9 |
| 256 | 64 | 90 | recovered | 17 |
| 256 | 64 | 95 | no_degradation | 5 |
| 256 | 64 | 95 | not_confirmed | 3 |
| 256 | 64 | 95 | recovered | 20 |
| 512 | 16 | 90 | not_confirmed | 14 |
| 512 | 16 | 90 | recovered | 14 |
| 512 | 16 | 95 | no_degradation | 1 |
| 512 | 16 | 95 | not_confirmed | 11 |
| 512 | 16 | 95 | recovered | 16 |
| 512 | 64 | 90 | not_confirmed | 23 |
| 512 | 64 | 90 | recovered | 5 |
| 512 | 64 | 95 | not_confirmed | 18 |
| 512 | 64 | 95 | recovered | 10 |

## H2: покрытие 80%-интервала, L=512, H=16

| Набор | Зона d | Coverage [CI95] | MAE × | Ширина × |
| --- | --- | --- | --- | --- |
| все ячейки | все окна | 0.774 [0.772; 0.777] | 1.028 | 1.021 |
| все ячейки | до смены | 0.780 [0.774; 0.786] | 0.999 | 1.001 |
| все ячейки | смена в горизонте | 0.736 [0.720; 0.750] | 1.113 | 0.977 |
| все ячейки | 0–63 | 0.711 [0.699; 0.723] | 1.055 | 0.961 |
| все ячейки | 64–127 | 0.742 [0.732; 0.751] | 1.024 | 0.959 |
| все ячейки | 128–255 | 0.778 [0.772; 0.783] | 1.072 | 1.073 |
| все ячейки | 256–511 | 0.793 [0.789; 0.797] | 1.019 | 1.049 |
| все ячейки | ≥512 | 0.781 [0.776; 0.785] | 1.005 | 1.002 |
| без drift | все окна | 0.775 [0.772; 0.777] | 1.018 | 1.012 |
| без drift | до смены | 0.780 [0.774; 0.786] | 0.999 | 1.001 |
| без drift | смена в горизонте | 0.723 [0.706; 0.739] | 1.147 | 0.972 |
| без drift | 0–63 | 0.706 [0.692; 0.719] | 1.045 | 0.952 |
| без drift | 64–127 | 0.752 [0.741; 0.762] | 0.982 | 0.936 |
| без drift | 128–255 | 0.782 [0.776; 0.789] | 1.050 | 1.060 |
| без drift | 256–511 | 0.792 [0.788; 0.796] | 1.013 | 1.038 |
| без drift | ≥512 | 0.781 [0.775; 0.786] | 1.007 | 1.002 |
| change | все окна | 0.772 [0.769; 0.775] | 1.045 | 1.032 |
| change | до смены | 0.780 [0.773; 0.787] | 1.000 | 1.001 |
| change | смена в горизонте | 0.726 [0.707; 0.745] | 1.080 | 0.964 |
| change | 0–63 | 0.684 [0.669; 0.699] | 1.194 | 0.993 |
| change | 64–127 | 0.736 [0.724; 0.748] | 1.152 | 1.069 |
| change | 128–255 | 0.780 [0.773; 0.787] | 1.068 | 1.077 |
| change | 256–511 | 0.795 [0.790; 0.799] | 1.012 | 1.046 |
| change | ≥512 | 0.779 [0.773; 0.786] | 1.005 | 1.001 |
| drift | все окна | 0.772 [0.766; 0.778] | 1.071 | 1.062 |
| drift | до смены | 0.781 [0.767; 0.794] | 1.000 | 1.002 |
| drift | смена в горизонте | 0.796 [0.764; 0.825] | 0.967 | 0.996 |
| drift | 0–63 | 0.734 [0.710; 0.756] | 1.105 | 1.007 |
| drift | 64–127 | 0.698 [0.671; 0.726] | 1.244 | 1.074 |
| drift | 128–255 | 0.757 [0.743; 0.771] | 1.180 | 1.135 |
| drift | 256–511 | 0.797 [0.788; 0.806] | 1.050 | 1.102 |
| drift | ≥512 | 0.780 [0.769; 0.791] | 1.000 | 1.001 |
| anomaly | все окна | 0.783 [0.778; 0.788] | 0.939 | 0.949 |
| anomaly | до смены | 0.784 [0.770; 0.796] | 0.999 | 1.001 |
| anomaly | смена в горизонте | 0.707 [0.675; 0.741] | 1.451 | 1.003 |
| anomaly | 0–63 | 0.766 [0.742; 0.790] | 0.667 | 0.817 |
| anomaly | 64–127 | 0.808 [0.790; 0.825] | 0.552 | 0.583 |
| anomaly | 128–255 | 0.791 [0.778; 0.802] | 0.999 | 1.013 |
| anomaly | 256–511 | 0.782 [0.773; 0.791] | 1.025 | 1.015 |
| anomaly | ≥512 | 0.783 [0.773; 0.795] | 1.021 | 1.007 |
| none | все окна | 0.783 [0.771; 0.792] | 0.975 | 1.000 |
| none | до смены | 0.766 [0.738; 0.792] | 0.977 | 0.996 |
| none | смена в горизонте | 0.753 [0.609; 0.872] | 0.993 | 0.976 |
| none | 0–63 | 0.817 [0.779; 0.853] | 0.926 | 0.989 |
| none | 64–127 | 0.741 [0.689; 0.787] | 1.074 | 0.991 |
| none | 128–255 | 0.779 [0.755; 0.803] | 0.992 | 1.005 |
| none | 256–511 | 0.789 [0.767; 0.808] | 0.962 | 1.006 |
| none | ≥512 | 0.796 [0.775; 0.819] | 0.958 | 0.999 |

## H2: покрытие 80%-интервала, L=512, H=64

| Набор | Зона d | Coverage [CI95] | MAE × | Ширина × |
| --- | --- | --- | --- | --- |
| все ячейки | все окна | 0.780 [0.777; 0.782] | 1.068 | 1.051 |
| все ячейки | до смены | 0.793 [0.787; 0.800] | 0.994 | 1.001 |
| все ячейки | смена в горизонте | 0.729 [0.719; 0.741] | 1.114 | 0.964 |
| все ячейки | 0–63 | 0.692 [0.677; 0.707] | 1.186 | 1.005 |
| все ячейки | 64–127 | 0.724 [0.710; 0.738] | 1.209 | 1.058 |
| все ячейки | 128–255 | 0.779 [0.771; 0.787] | 1.134 | 1.121 |
| все ячейки | 256–511 | 0.816 [0.812; 0.820] | 1.023 | 1.089 |
| все ячейки | ≥512 | 0.796 [0.791; 0.801] | 0.997 | 1.004 |
| без drift | все окна | 0.782 [0.779; 0.785] | 1.056 | 1.043 |
| без drift | до смены | 0.793 [0.786; 0.800] | 0.994 | 1.002 |
| без drift | смена в горизонте | 0.715 [0.703; 0.727] | 1.145 | 0.958 |
| без drift | 0–63 | 0.694 [0.678; 0.710] | 1.174 | 1.005 |
| без drift | 64–127 | 0.744 [0.729; 0.758] | 1.160 | 1.053 |
| без drift | 128–255 | 0.791 [0.782; 0.798] | 1.096 | 1.107 |
| без drift | 256–511 | 0.816 [0.811; 0.820] | 1.013 | 1.075 |
| без drift | ≥512 | 0.797 [0.791; 0.803] | 0.997 | 1.004 |
| change | все окна | 0.777 [0.773; 0.781] | 1.075 | 1.052 |
| change | до смены | 0.794 [0.785; 0.802] | 0.994 | 1.002 |
| change | смена в горизонте | 0.710 [0.696; 0.724] | 1.126 | 0.944 |
| change | 0–63 | 0.662 [0.643; 0.679] | 1.285 | 0.993 |
| change | 64–127 | 0.727 [0.708; 0.745] | 1.242 | 1.107 |
| change | 128–255 | 0.785 [0.775; 0.795] | 1.131 | 1.132 |
| change | 256–511 | 0.818 [0.813; 0.823] | 1.015 | 1.085 |
| change | ≥512 | 0.795 [0.789; 0.801] | 0.998 | 1.001 |
| drift | все окна | 0.769 [0.761; 0.776] | 1.125 | 1.087 |
| drift | до смены | 0.793 [0.776; 0.809] | 0.996 | 1.000 |
| drift | смена в горизонте | 0.797 [0.779; 0.814] | 0.979 | 0.993 |
| drift | 0–63 | 0.684 [0.653; 0.715] | 1.248 | 1.001 |
| drift | 64–127 | 0.631 [0.592; 0.668] | 1.463 | 1.078 |
| drift | 128–255 | 0.727 [0.705; 0.747] | 1.332 | 1.190 |
| drift | 256–511 | 0.819 [0.808; 0.829] | 1.075 | 1.159 |
| drift | ≥512 | 0.794 [0.783; 0.804] | 0.997 | 1.005 |
| anomaly | все окна | 0.797 [0.792; 0.802] | 1.005 | 1.021 |
| anomaly | до смены | 0.795 [0.780; 0.810] | 0.993 | 1.002 |
| anomaly | смена в горизонте | 0.720 [0.692; 0.749] | 1.246 | 1.000 |
| anomaly | 0–63 | 0.792 [0.770; 0.813] | 0.879 | 1.053 |
| anomaly | 64–127 | 0.802 [0.784; 0.821] | 0.930 | 0.895 |
| anomaly | 128–255 | 0.809 [0.797; 0.820] | 0.999 | 1.037 |
| anomaly | 256–511 | 0.809 [0.799; 0.818] | 1.012 | 1.054 |
| anomaly | ≥512 | 0.802 [0.789; 0.815] | 1.003 | 1.014 |
| none | все окна | 0.796 [0.786; 0.805] | 0.979 | 0.998 |
| none | до смены | 0.779 [0.733; 0.820] | 0.993 | 0.997 |
| none | смена в горизонте | 0.787 [0.747; 0.822] | 0.974 | 0.991 |
| none | 0–63 | 0.790 [0.760; 0.817] | 0.985 | 0.993 |
| none | 64–127 | 0.768 [0.717; 0.809] | 1.056 | 0.989 |
| none | 128–255 | 0.799 [0.768; 0.824] | 0.977 | 1.008 |
| none | 256–511 | 0.802 [0.781; 0.821] | 0.973 | 1.002 |
| none | ≥512 | 0.809 [0.778; 0.843] | 0.947 | 0.994 |

## Аудит и файлы

`reference_audit.csv`: эталоны и число окон по каждому ряду/L/H/метрике. `exclusion_audit.csv`: объём исключений. `thresholds.csv`: сырой и ограниченный порог.
`curves.csv.gz`: описательные кривые/CI и n_valid; `d_star.csv`: все статусы, частоты бутстрепа, условные CI и d*/L. `coverage_zones.csv`, `coverage_zones_no_drift.csv`: H2.
Исключено отношений (по метрикам суммарно): 0. Окна перекрываются; это не число независимых наблюдений.
Время расчёта: 43.4 с (только кеш, без инференса).
