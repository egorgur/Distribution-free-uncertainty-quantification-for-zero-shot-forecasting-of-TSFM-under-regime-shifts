"""Обёртки TSFM: единый интерфейс «контексты → квантильный прогноз».

Поддержаны Chronos-Bolt (основная модель работы) и Chronos-2 — обе из пакета
chronos-forecasting. Создавать модель через load_forecaster(model_id).
Эмбеддинги патчей для S5 (encode_patches) — у обеих моделей: вход энкодера и выход
последнего слоя. Эмбеддинги всех уровней энкодера для этапа B, H4 (encode_layers) — тоже
у обеих: промежуточные слои пакет не отдаёт, они снимаются хуками на блоки энкодера.

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

import contextlib
import math

import numpy as np
import torch
from chronos import BaseChronosPipeline
from einops import rearrange

# Квантили, которые отдают все обёртки (у Chronos-Bolt это все её квантили).
QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


@contextlib.contextmanager
def capture_block_outputs(blocks: torch.nn.ModuleList, n_patch: int):
    """Пока контекст открыт, выход каждого блока энкодера на патчах контекста добавляется
    в список (по одному тензору (батч, n_patch, d_model) на блок, по порядку вызова).

    У обеих моделей скрытые состояния — первый элемент выхода блока: у T5Block (Bolt) это
    кортеж, у Chronos2EncoderBlock — ModelOutput с полем hidden_states.
    """
    captured: list[torch.Tensor] = []
    hooks = [b.register_forward_hook(lambda _m, _inp, out: captured.append(out[0][:, :n_patch])) for b in blocks]
    try:
        yield captured
    finally:
        for h in hooks:
            h.remove()


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
    # True, если эмбеддинги патчей контекста зависят от горизонта прогноза. Тогда S5
    # считается отдельно для каждого горизонта — из того же прохода, что и прогноз.
    embeddings_depend_on_horizon = False

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

    def encode_patches(
        self, contexts: np.ndarray, batch_size: int = 256, horizon: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Эмбеддинги патчей контекста (S5): (вход энкодера, выход последнего слоя)."""
        raise NotImplementedError(f"{self.model_id}: эмбеддинги пока не поддержаны")

    @property
    def n_blocks(self) -> int:
        """Число блоков (слоёв) энкодера."""
        return len(self.pipeline.model.encoder.block)

    @property
    def layer_names(self) -> list[str]:
        """Уровни encode_layers по порядку: input, block1 … blockN, final."""
        return ["input"] + [f"block{i + 1}" for i in range(self.n_blocks)] + ["final"]

    @property
    def layer_depths(self) -> list[float]:
        """Нормированная глубина уровней: input — 0, блок i — i / N, final — 1 (как блок N)."""
        n = self.n_blocks
        return [0.0] + [(i + 1) / n for i in range(n)] + [1.0]

    def encode_layers(self, contexts: np.ndarray, batch_size: int = 256, horizon: int | None = None) -> np.ndarray:
        """Эмбеддинги патчей контекста на всех уровнях энкодера (этап B, H4).

        contexts: (окна, L) без NaN, L кратно 16. Возвращает (окна, уровни, L / 16, d_model),
        float32; уровни — layer_names: вход энкодера, выход каждого блока, выход последнего блока
        после итоговой layer norm (= encode_patches()[1]). Патч i — точки 16·i … 16·i + 15 контекста.
        """
        raise NotImplementedError(f"{self.model_id}: эмбеддинги пока не поддержаны")

    def _check_encode_input(self, contexts: np.ndarray) -> np.ndarray:
        contexts = np.asarray(contexts, dtype=np.float32)
        L = contexts.shape[1]
        if L % self.patch_size or L > self.max_context or np.isnan(contexts).any():
            raise ValueError(f"нужны контексты без NaN длины, кратной {self.patch_size}; получено L = {L}")
        return contexts


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
    def encode_patches(
        self, contexts: np.ndarray, batch_size: int = 256, horizon: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Эмбеддинги патчей контекста: (вход энкодера, выход последнего слоя энкодера).

        contexts: (окна, L) без NaN, L кратно длине патча (16) и не больше max_context.
        horizon: не нужен — энкодер Bolt будущего не видит (горизонт задаёт только декодер).
        Возвращает два массива (окна, L / 16, d_model), float32. Патч i покрывает точки
        16·i … 16·i + 15 контекста, последний кончается в t. Токен [REG] отброшен.
        - вход энкодера — эмбеддинг каждого патча по отдельности (после нормировки по всему
          контексту), до трансформера;
        - выход последнего слоя (после итоговой layer norm) — то, что читает декодер; из-за
          внимания каждый токен зависит от всего контекста.
        Это тот же проход энкодера, что и в прогнозе, поэтому в работе S5 не требует доп. прогонов.
        """
        contexts = self._check_encode_input(contexts)
        n, L = contexts.shape
        model = self.pipeline.model
        n_patch = L // self.patch_size
        inputs, finals = [], []
        for start in range(0, n, batch_size):
            x = torch.as_tensor(contexts[start : start + batch_size], device=model.device)
            final, _, input_embeds, _ = model.encode(context=x)
            inputs.append(input_embeds[:, :n_patch].float().cpu().numpy())
            finals.append(final[:, :n_patch].float().cpu().numpy())
        return np.concatenate(inputs), np.concatenate(finals)

    @torch.no_grad()
    def encode_layers(self, contexts: np.ndarray, batch_size: int = 256, horizon: int | None = None) -> np.ndarray:
        """См. ChronosForecaster.encode_layers. horizon не нужен: энкодер Bolt будущего не видит.

        Тот же проход, что и в прогнозе (model.encode); выходы блоков снимаются хуками.
        """
        contexts = self._check_encode_input(contexts)
        model = self.pipeline.model
        n_patch = contexts.shape[1] // self.patch_size
        out = []
        for start in range(0, len(contexts), batch_size):
            x = torch.as_tensor(contexts[start : start + batch_size], device=model.device)
            with capture_block_outputs(model.encoder.block, n_patch) as blocks:
                final, _, input_embeds, _ = model.encode(context=x)
            assert len(blocks) == self.n_blocks, (len(blocks), self.n_blocks)
            levels = [input_embeds[:, :n_patch], *blocks, final[:, :n_patch]]
            out.append(torch.stack(levels, dim=1).float().cpu().numpy())
        return np.concatenate(out)


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

    # Внимание по времени двунаправленное, а после патчей контекста и токена [REG] модель
    # дописывает токены будущего — по одному на 16 шагов горизонта. Поэтому выход энкодера
    # на патчах контекста зависит от горизонта (вход энкодера — нет).
    embeddings_depend_on_horizon = True

    def encode_patches(
        self, contexts: np.ndarray, batch_size: int = 256, horizon: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Эмбеддинги патчей контекста: (вход энкодера, выход последнего слоя энкодера).

        contexts: (окна, L) без NaN, L кратно длине патча (16) и не больше max_context.
        horizon: обязателен — от него зависит число токенов будущего в том же проходе.
        Возвращает два массива (окна, L / 16, d_model), float32. Патч i покрывает точки
        16·i … 16·i + 15 контекста, последний кончается в t. Токены [REG] и будущего отброшены.
        - вход энкодера — эмбеддинг каждого патча по отдельности (нормировка по всему контексту,
          arcsinh, кодировка времени), до трансформера; от горизонта не зависит;
        - выход последнего слоя (после итоговой layer norm) — из того же прохода, что и прогноз
          на horizon шагов; из-за внимания каждый токен зависит от всего контекста и от числа
          токенов будущего.
        """
        inputs, finals, _ = self._encode(contexts, horizon, batch_size)
        return inputs, finals

    def encode_patches_with_forecast(
        self, contexts: np.ndarray, horizon: int, batch_size: int = 256
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """То же, что encode_patches, плюс квантильный прогноз (окна, 9, horizon) из того же прохода.
        Нужен для проверки: прогноз должен совпасть с predict_quantiles, то есть эмбеддинги взяты
        именно из прохода прогноза."""
        return self._encode(contexts, horizon, batch_size)

    @torch.no_grad()
    def _encode(
        self, contexts: np.ndarray, horizon: int | None, batch_size: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        contexts = self._check_encode_input(contexts)
        n_out = self._n_output_patches(horizon)
        inputs, finals, quantiles = [], [], []
        for start in range(0, len(contexts), batch_size):
            inp, fin, q = self._encode_batch(contexts[start : start + batch_size], n_out, horizon)
            inputs.append(inp.float().cpu().numpy())
            finals.append(fin.float().cpu().numpy())
            quantiles.append(q.float().cpu().numpy())
        return np.concatenate(inputs), np.concatenate(finals), np.concatenate(quantiles)

    def encode_layers(self, contexts: np.ndarray, batch_size: int = 256, horizon: int | None = None) -> np.ndarray:
        """См. ChronosForecaster.encode_layers. horizon обязателен: выход энкодера от него зависит."""
        return self.encode_layers_with_forecast(contexts, horizon, batch_size)[0]

    @torch.no_grad()
    def encode_layers_with_forecast(
        self, contexts: np.ndarray, horizon: int, batch_size: int = 256
    ) -> tuple[np.ndarray, np.ndarray]:
        """encode_layers плюс квантильный прогноз (окна, 9, horizon) из того же прохода — для проверки,
        что уровни сняты именно с прохода прогноза."""
        contexts = self._check_encode_input(contexts)
        n_out = self._n_output_patches(horizon)
        n_patch = contexts.shape[1] // self.patch_size
        layers, quantiles = [], []
        for start in range(0, len(contexts), batch_size):
            with capture_block_outputs(self.pipeline.model.encoder.block, n_patch) as blocks:
                inp, fin, q = self._encode_batch(contexts[start : start + batch_size], n_out, horizon)
            assert len(blocks) == self.n_blocks, (len(blocks), self.n_blocks)
            layers.append(torch.stack([inp, *blocks, fin], dim=1).float().cpu().numpy())
            quantiles.append(q.float().cpu().numpy())
        return np.concatenate(layers), np.concatenate(quantiles)

    def _n_output_patches(self, horizon: int | None) -> int:
        if horizon is None or not 0 < horizon <= self.max_horizon:
            raise ValueError(f"нужен горизонт 1…{self.max_horizon}, получено {horizon}")
        # Как в Chronos2Pipeline._predict_batch.
        return math.ceil(horizon / self.pipeline.model.chronos_config.output_patch_size)

    def _encode_batch(
        self, contexts: np.ndarray, n_out: int, horizon: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Один проход энкодера: (вход энкодера, выход после итоговой layer norm) на патчах контекста
        и квантили (батч, 9, horizon) из того же прохода."""
        model = self.pipeline.model
        n_patch = contexts.shape[1] // self.patch_size
        out_patch = model.chronos_config.output_patch_size
        x = torch.as_tensor(contexts, device=model.device)
        # Каждый ряд — своя группа: обмена между рядами батча нет (cross_learning выключен).
        group_ids = torch.arange(len(x), device=model.device)
        enc, loc_scale, _, n_ctx = model.encode(context=x, group_ids=group_ids, num_output_patches=n_out)
        assert n_ctx == n_patch, (n_ctx, n_patch)
        hidden = enc.last_hidden_state  # (батч, патчи контекста + [REG] + n_out, d_model)
        # Вход энкодера: тот же расчёт, что внутри model.encode, до трансформера.
        patched, _, _ = model._prepare_patched_context(x)
        inputs = model.input_patch_embedding(patched)
        # Прогноз из того же прохода — как в Chronos2Model.forward.
        q = model.output_patch_embedding(hidden[:, -n_out:])
        q = rearrange(q, "b n (q p) -> b (q n p)", q=model.num_quantiles, p=out_patch)
        q = rearrange(model.instance_norm.inverse(q, loc_scale), "b (q h) -> b q h", q=model.num_quantiles)
        return inputs, hidden[:, :n_patch], q[:, self._idx, :horizon]
