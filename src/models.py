"""Обёртки TSFM: единый интерфейс «контексты → квантильный прогноз».

Поддержаны Chronos-Bolt (основная модель работы) и Chronos-2 — обе из пакета
chronos-forecasting. Создавать модель через load_forecaster(model_id).
Эмбеддинги патчей для S5 (encode_patches) — пока только у Chronos-Bolt: вход энкодера и выход
последнего слоя. Эмбеддинги всех слоёв (этап B, H4) добавим, когда до них дойдём.

Общее (chronos-forecasting 2.3.2, проверено по исходникам и config.json):
- квантили детерминированы, без сэмплирования;
- ряд режется на патчи по 16 точек; патчи выравниваются по ПРАВОМУ краю
  (последний патч всегда заканчивается в момент t), неполный патч дополняется NaN слева;
- контекст нормируется по всему контексту (среднее и std);
- контексты разной длины в одном батче выравниваются NaN слева;
- прогноз дальше одного прохода пакет достраивает авторегрессией — мы это запрещаем.

Различия:
- Chronos-Bolt: энкодер-декодер (T5), 9 квантилей 0.1…0.9, один проход — до 64 шагов.
- Chronos-2: только энкодер, 21 квантиль 0.01…0.99 (мы отдаём 0.1…0.9, чтобы модели
  были сравнимы), один проход — до 1024 шагов, после нормировки применяется arcsinh.
  Умеет обмениваться информацией между рядами батча (cross_learning) — выключено,
  иначе прогноз окна зависел бы от соседей по батчу.
"""

from __future__ import annotations

import numpy as np
import torch
from chronos import BaseChronosPipeline

# Квантили, которые отдают все обёртки (у Chronos-Bolt это все её квантили).
QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def load_forecaster(model_id: str, device: str = "cpu") -> "ChronosForecaster":
    """Обёртка по имени весов на Hugging Face, например amazon/chronos-bolt-small."""
    if "chronos-bolt" in model_id:
        return ChronosBoltForecaster(model_id, device)
    if "chronos-2" in model_id:
        return Chronos2Forecaster(model_id, device)
    raise ValueError(f"неизвестная модель: {model_id}")


class ChronosForecaster:
    """Общая часть: загрузка, проверка входа, разбиение на батчи.

    Пример:
        model = load_forecaster("amazon/chronos-bolt-small")
        q = model.predict_quantiles([x[:512], x[:256]], horizon=16)
        q.shape  # (2, 9, 16): ряд × квантиль × шаг горизонта
        median = q[:, model.quantile_levels.index(0.5), :]
    """

    quantile_levels = QUANTILE_LEVELS
    # True, если прогноз на h шагов совпадает с началом прогноза на max_horizon шагов.
    # Тогда короткий горизонт не нужно считать отдельно.
    short_horizon_is_prefix = False

    def __init__(self, model_id: str, device: str = "cpu"):
        self.model_id = model_id
        self.pipeline = BaseChronosPipeline.from_pretrained(model_id, device_map=device)
        self.pipeline.model.eval()  # режим инференса: выключает dropout

    @property
    def max_horizon(self) -> int:
        """Максимальный горизонт за один проход."""
        return self.pipeline.model_prediction_length

    @property
    def patch_size(self) -> int:
        return self.pipeline.model.chronos_config.input_patch_size

    @property
    def max_context(self) -> int:
        """Более длинный контекст модель молча обрежет до этой длины."""
        return self.pipeline.model_context_length

    def predict_quantiles(
        self,
        contexts: list[np.ndarray],
        horizon: int,
        batch_size: int = 256,
    ) -> np.ndarray:
        """Квантильные прогнозы для набора контекстов.

        contexts: список одномерных массивов, длины могут различаться
            (так удобно считать S4 и S7 одним вызовом). NaN = пропуск.
        horizon: число шагов прогноза, не больше max_horizon.
        Возвращает массив (len(contexts), 9 квантилей, horizon), float32.
        """
        if horizon > self.max_horizon:
            # Пакет умеет прогнозировать дальше авторегрессией, но такие интервалы
            # систематически переуверены (Adler et al., ICLR 2026) — запрещаем явно.
            raise ValueError(f"horizon={horizon} > max_horizon={self.max_horizon}")
        for i, c in enumerate(contexts):
            if np.ndim(c) != 1 or len(c) == 0:
                raise ValueError(f"контекст {i}: ожидается непустой одномерный массив")
            if len(c) > self.max_context:
                raise ValueError(f"контекст {i}: длина {len(c)} > max_context={self.max_context}")

        out = []
        for start in range(0, len(contexts), batch_size):
            batch = [torch.as_tensor(c, dtype=torch.float32) for c in contexts[start : start + batch_size]]
            out.append(self._predict_batch(batch, horizon))
        return np.concatenate(out, axis=0)

    def _predict_batch(self, batch: list[torch.Tensor], horizon: int) -> np.ndarray:
        raise NotImplementedError

    def encode_patches(self, contexts: np.ndarray, batch_size: int = 256) -> tuple[np.ndarray, np.ndarray]:
        """Эмбеддинги патчей контекста (S5). Реализовано только у Chronos-Bolt."""
        raise NotImplementedError(f"{self.model_id}: эмбеддинги пока не поддержаны")


class ChronosBoltForecaster(ChronosForecaster):
    """Замороженная Chronos-Bolt (tiny / mini / small / base)."""

    # Декодер выдаёт все 64 шага за один проход, пакет лишь обрезает их до horizon
    # (проверено: разница 0 для H = 16 и 64).
    short_horizon_is_prefix = True

    def __init__(self, model_id: str = "amazon/chronos-bolt-small", device: str = "cpu"):
        super().__init__(model_id, device)
        assert list(self.pipeline.quantiles) == QUANTILE_LEVELS, self.pipeline.quantiles

    def _predict_batch(self, batch: list[torch.Tensor], horizon: int) -> np.ndarray:
        # Выход (батч, 9, horizon).
        q = self.pipeline.predict(batch, prediction_length=horizon, limit_prediction_length=True)
        return q.numpy()

    @torch.no_grad()
    def encode_patches(self, contexts: np.ndarray, batch_size: int = 256) -> tuple[np.ndarray, np.ndarray]:
        """Эмбеддинги патчей контекста: (вход энкодера, выход последнего слоя энкодера).

        contexts: (окна, L) без NaN, L кратно длине патча (16) и не больше max_context.
        Возвращает два массива (окна, L / 16, d_model), float32. Патч i покрывает точки
        16·i … 16·i + 15 контекста, последний кончается в t. Токен [REG] отброшен.
        - вход энкодера — эмбеддинг каждого патча по отдельности (после нормировки по всему
          контексту), до трансформера;
        - выход последнего слоя (после итоговой layer norm) — то, что читает декодер; из-за
          внимания каждый токен зависит от всего контекста.
        Это тот же проход энкодера, что и в прогнозе, поэтому в работе S5 не требует доп. прогонов.
        """
        contexts = np.asarray(contexts, dtype=np.float32)
        n, L = contexts.shape
        if L % self.patch_size or L > self.max_context or np.isnan(contexts).any():
            raise ValueError(f"нужны контексты без NaN длины, кратной {self.patch_size}; получено L = {L}")
        model = self.pipeline.model
        n_patch = L // self.patch_size
        inputs, finals = [], []
        for start in range(0, n, batch_size):
            x = torch.as_tensor(contexts[start : start + batch_size], device=model.device)
            final, _, input_embeds, _ = model.encode(context=x)
            inputs.append(input_embeds[:, :n_patch].float().cpu().numpy())
            finals.append(final[:, :n_patch].float().cpu().numpy())
        return np.concatenate(inputs), np.concatenate(finals)


class Chronos2Forecaster(ChronosForecaster):
    """Замороженная Chronos-2, одномерный режим без обмена информацией между рядами.

    Число запрошенных патчей будущего влияет на весь прогноз: прогноз на 16 шагов
    отличается от первых 16 шагов прогноза на 64 (в среднем на 0.006 std ряда,
    максимум 0.03), поэтому каждый горизонт считается отдельным вызовом.
    """

    def __init__(self, model_id: str = "amazon/chronos-2", device: str = "cpu"):
        super().__init__(model_id, device)
        levels = [round(q, 4) for q in self.pipeline.quantiles]
        self._idx = [levels.index(q) for q in QUANTILE_LEVELS]

    def _predict_batch(self, batch: list[torch.Tensor], horizon: int) -> np.ndarray:
        # Список из len(batch) тензоров (1, 21, horizon) → (батч, 9, horizon).
        preds = self.pipeline.predict(
            batch,
            prediction_length=horizon,
            batch_size=len(batch),
            cross_learning=False,
            limit_prediction_length=True,
        )
        q = torch.stack([p[0] for p in preds])
        return q[:, self._idx, :].numpy()
