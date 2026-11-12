# MCC5: черновик запроса авторам

**Не отправлено.** Перед отправкой заполнить подпись, проверить получателей
и решить, обращаться к одному автору или к обоим. Это просьба о документации,
не утверждение ошибки авторов и не запрос на одобрение наших результатов.

Проверенные контакты и источники — `docs/results/mcc5_source_followup.md`:

- Xiao He: `hexiao@tsinghua.edu.cn` (corresponding author статьи).
- Zeyi Liu: `liuzy21@mails.tsinghua.edu.cn` (персональная страница соавтора).

## Текст письма

**Subject:** MCC5-THU Motor: data version, torque scaling and settled intervals

Dear Prof. He and Mr. Liu,

Thank you for releasing the MCC5-THU Motor dataset. I am studying the reliability
of frozen time-series forecasting models around changes in operating conditions.
Before running forecasting experiments, I would like to clarify the recording
units and whether physically settled reference intervals are available.

I inspected one complete file from the linked Hugging Face mirror, commit
`897bc5d2504980418249f98e733bfd7f61056186`:

`MCC5-THU Motor_torque_circulation/health_torque_circulation_20Nm_1000rpm_250702154813d.csv`

It contains 1,152,000 rows, nine columns including time, and 90 seconds at
12.8 kHz. Its uncompressed SHA256 is
`f331e955f447d5511534e0f4b664b3df620321d31e82850e9c58a7b4466e2a7a`.

Could you please clarify the following?

1. **Version and units.** The early preprint and GitHub README describe eight
   columns with torque in Nm, whereas arXiv v2 / the published article describe
   nine columns of raw voltages and a torque sensitivity of 100 mV/Nm. Which
   description applies to this file? What gain, offset and conversion should
   be applied to column 3? Its mean is about 0.2181 on 10–30 s and 0.2197 on
   60–80 s, corresponding to the example 20 Nm command. Is the commanded torque
   defined at the same shaft as the measurement? What changed between the
   Mendeley V1 and V2 archives, and which version does the mirror represent?
2. **Command and settling.** Does the Fig.2 schedule (20 Nm on 10–30/60–80 s,
   15 Nm on 35–55 s) apply to this exact recording? Are command timestamps,
   independently documented settling times, acceptable residual drift, or
   longer constant-condition recordings available? We do not want to assume
   that every constant command is already a settled measurement.
3. **Independent repeats.** Are there independent recordings of the same
   healthy condition and command profile, rather than different faults/speeds
   or repeated plateaus within one run? If so, how are repeat IDs and shared
   acquisition sessions identified?
4. **Timing.** How many key-phase pulses correspond to one motor revolution?
   Are the stored times measured acquisition timestamps or an index-derived
   nominal grid, and are dropped samples recorded separately?

A calibration note, acquisition/export script, or pointers to the relevant
documentation would be very helpful. No new experiment is requested. I have
not assumed that the observed differences invalidate the data, and have not
used a selected late tail as a stable reference.

Thank you for your time and for making the data available.

Best regards,
[Name]
[Affiliation, if applicable]

## После получения ответа

Сохранить ответ с датой/ссылкой и явно отделить слова авторов от собственной
интерпретации. Личную переписку не публиковать без согласия; для отчёта
использовать согласованную цитату или публичный источник. Не переносить ответ
на все версии/файлы без подтверждения и не менять протокол автоматически.