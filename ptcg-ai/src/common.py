from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from base_data import (
    CATEGORICAL_FIELDS,
    FLAG_FIELDS,
    NUMERIC_FIELDS,
    DecisionBatch,
    TokenBatch,
)


@dataclass(frozen=True)
class ModelConfig:
    d_model: int = 192
    nhead: int = 6
    layers: int = 4
    dim_feedforward: int = 768
    dropout: float = 0.1
    categorical_vocab_sizes: tuple[int, ...] = (
        16,
        16,
        16,
        4,
        4,
        2048,
        2048,
        256,
        256,
        256,
        256,
        16,
        64,
        32,
        32,
        2048,
        16,
        32,
        16,
    )


class TokenEmbedding(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        """Create embeddings for the unified token fields."""
        super().__init__()
        self.categorical = nn.ModuleList(
            nn.Embedding(size, config.d_model, padding_idx=0)
            for size in config.categorical_vocab_sizes
        )
        self.numeric = nn.Linear(len(NUMERIC_FIELDS), config.d_model, bias=False)
        self.flags = nn.Linear(len(FLAG_FIELDS), config.d_model, bias=False)
        self.normalization = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, batch: TokenBatch) -> torch.Tensor:
        """Embed one padded token batch."""
        embedded = self.numeric(batch.numeric) + self.flags(batch.flags)
        for index, table in enumerate(self.categorical):
            values = batch.categorical[..., index].clamp_max(table.num_embeddings - 1)
            embedded = embedded + table(values)
        return self.dropout(self.normalization(embedded))


class EntityPointerPolicy(nn.Module):
    def __init__(self, config: ModelConfig | None = None) -> None:
        """Create the shared base and expert policy model."""
        super().__init__()
        self.config = config or ModelConfig()
        layer = nn.TransformerEncoderLayer(
            d_model=self.config.d_model,
            nhead=self.config.nhead,
            dim_feedforward=self.config.dim_feedforward,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.embedding = TokenEmbedding(self.config)
        self.encoder = nn.TransformerEncoder(
            layer,
            self.config.layers,
            nn.LayerNorm(self.config.d_model),
            enable_nested_tensor=False,
        )
        self.initial_state = nn.Linear(self.config.d_model, self.config.d_model)
        self.decoder = nn.GRUCell(self.config.d_model, self.config.d_model)
        self.query = nn.Linear(self.config.d_model, self.config.d_model, bias=False)
        self.key = nn.Linear(self.config.d_model, self.config.d_model, bias=False)
        self.value = nn.Linear(self.config.d_model, self.config.d_model, bias=False)
        self.stop_key = nn.Parameter(torch.empty(self.config.d_model))
        self.stop_value = nn.Parameter(torch.empty(self.config.d_model))
        nn.init.normal_(self.stop_key, std=self.config.d_model**-0.5)
        nn.init.normal_(self.stop_value, std=self.config.d_model**-0.5)

    def _encode(
        self,
        batch: DecisionBatch,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode state and option tokens."""
        state_width = batch.state.mask.shape[1]
        tokens = torch.cat((self.embedding(batch.state), self.embedding(batch.options)), dim=1)
        mask = torch.cat((batch.state.mask, batch.options.mask), dim=1)
        encoded = self.encoder(tokens, src_key_padding_mask=~mask)
        summary = torch.tanh(self.initial_state(encoded[:, 0]))
        options = encoded[:, state_width:]
        return summary, self.key(options), self.value(options)

    def _logits(
        self,
        hidden: torch.Tensor,
        keys: torch.Tensor,
        option_mask: torch.Tensor,
        selected: torch.Tensor,
        counts: torch.Tensor,
        minimum: torch.Tensor,
        maximum: torch.Tensor,
    ) -> torch.Tensor:
        """Score legal options and the stop action."""
        query = self.query(hidden)
        option_logits = torch.einsum("bd,bod->bo", query, keys) / sqrt(self.config.d_model)
        stop_logits = torch.einsum("bd,d->b", query, self.stop_key).unsqueeze(1)
        logits = torch.cat((option_logits, stop_logits), dim=1)
        option_valid = option_mask & ~selected & (counts < maximum).unsqueeze(1)
        stop_valid = (counts >= minimum).unsqueeze(1)
        valid = torch.cat((option_valid, stop_valid), dim=1)
        return logits.masked_fill(~valid, torch.finfo(logits.dtype).min)

    def _advance(
        self,
        hidden: torch.Tensor,
        values: torch.Tensor,
        selected: torch.Tensor,
        counts: torch.Tensor,
        choice: torch.Tensor,
        active: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Advance the decoder after one option or stop choice."""
        option_count = values.shape[1]
        chooses_option = active & (choice < option_count)
        safe_choice = choice.clamp(0, max(option_count - 1, 0))
        if option_count:
            chosen = values.gather(
                1,
                safe_choice[:, None, None].expand(-1, 1, values.shape[-1]),
            ).squeeze(1)
            selected = selected | (
                F.one_hot(safe_choice, option_count).bool() & chooses_option.unsqueeze(1)
            )
        else:
            chosen = torch.zeros_like(hidden)
        decoder_input = torch.where(chooses_option.unsqueeze(1), chosen, self.stop_value)
        updated = self.decoder(decoder_input, hidden)
        hidden = torch.where(active.unsqueeze(1), updated, hidden)
        return hidden, selected, counts + chooses_option.long(), chooses_option

    def initial_logits(self, batch: DecisionBatch) -> torch.Tensor:
        """Return scores for the first selection step."""
        hidden, keys, _ = self._encode(batch)
        selected = torch.zeros_like(batch.options.mask)
        counts = torch.zeros_like(batch.minimum)
        return self._logits(
            hidden,
            keys,
            batch.options.mask,
            selected,
            counts,
            batch.minimum,
            batch.maximum,
        )

    def teacher_forced_logits(self, batch: DecisionBatch) -> torch.Tensor:
        """Return scores for every labeled selection step."""
        if batch.targets is None:
            raise ValueError("teacher forcing requires labeled decisions")
        hidden, keys, values = self._encode(batch)
        selected = torch.zeros_like(batch.options.mask)
        counts = torch.zeros_like(batch.minimum)
        outputs = []
        for target in batch.targets.unbind(1):
            outputs.append(
                self._logits(
                    hidden,
                    keys,
                    batch.options.mask,
                    selected,
                    counts,
                    batch.minimum,
                    batch.maximum,
                )
            )
            hidden, selected, counts, _ = self._advance(
                hidden,
                values,
                selected,
                counts,
                target,
                target >= 0,
            )
        return torch.stack(outputs, dim=1)

    def loss(self, batch: DecisionBatch) -> torch.Tensor:
        """Calculate behavior-cloning cross entropy."""
        if batch.targets is None:
            raise ValueError("loss requires labeled decisions")
        logits = self.teacher_forced_logits(batch)
        return F.cross_entropy(
            logits.flatten(0, 1),
            batch.targets.flatten(),
            ignore_index=-100,
        )

    @torch.inference_mode()
    def decode(self, batch: DecisionBatch) -> list[list[int]]:
        """Decode legal non-repeating option selections."""
        hidden, keys, values = self._encode(batch)
        selected = torch.zeros_like(batch.options.mask)
        counts = torch.zeros_like(batch.minimum)
        finished = torch.zeros_like(batch.minimum, dtype=torch.bool)
        actions = [[] for _ in range(len(batch.minimum))]

        for _ in range(int(batch.maximum.max().item()) + 1):
            logits = self._logits(
                hidden,
                keys,
                batch.options.mask,
                selected,
                counts,
                batch.minimum,
                batch.maximum,
            )
            choice = logits.argmax(dim=1)
            active = ~finished
            hidden, selected, counts, chooses_option = self._advance(
                hidden,
                values,
                selected,
                counts,
                choice,
                active,
            )
            for row, index in enumerate(choice.tolist()):
                if chooses_option[row]:
                    actions[row].append(index)
            finished = finished | (active & ~chooses_option) | (counts >= batch.maximum)
            if finished.all():
                break
        return actions

    def forward(self, batch: DecisionBatch) -> torch.Tensor:
        """Return first-step action scores."""
        return self.initial_logits(batch)


@dataclass(frozen=True)
class PolicyOutput:
    actions: list[list[int]]
    log_probability: torch.Tensor
    entropy: torch.Tensor
    value: torch.Tensor
    logits: torch.Tensor | None = None


class ActorCriticPolicy(nn.Module):
    def __init__(self, policy: EntityPointerPolicy) -> None:
        """Add a small value head without changing the published policy format."""
        super().__init__()
        self.policy = policy
        self.value_head = nn.Linear(policy.config.d_model, 1)
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)

    def _sequence(self, batch: DecisionBatch, sample: bool) -> PolicyOutput:
        """Sample or rescore one complete legal CABT action per decision."""
        hidden, keys, values = self.policy._encode(batch)
        state_value = self.value_head(hidden).squeeze(1)
        selected = torch.zeros_like(batch.options.mask)
        counts = torch.zeros_like(batch.minimum)
        finished = torch.zeros_like(batch.minimum, dtype=torch.bool)
        actions = [[] for _ in range(len(batch.minimum))]
        log_probability = torch.zeros_like(state_value, dtype=torch.float32)
        entropy = torch.zeros_like(log_probability)
        entropy_steps = torch.zeros_like(log_probability)
        outputs = []
        steps = (
            batch.targets.shape[1]
            if batch.targets is not None
            else int(batch.maximum.max().item()) + 1
        )

        for step in range(steps):
            logits = self.policy._logits(
                hidden,
                keys,
                batch.options.mask,
                selected,
                counts,
                batch.minimum,
                batch.maximum,
            )
            distribution = torch.distributions.Categorical(logits=logits.float())
            if batch.targets is None:
                active = ~finished
                choice = distribution.sample() if sample else logits.argmax(dim=1)
            else:
                target = batch.targets[:, step]
                active = target >= 0
                choice = target.clamp_min(0)
                outputs.append(logits)

            log_probability += torch.where(
                active,
                distribution.log_prob(choice),
                torch.zeros_like(log_probability),
            )
            entropy += torch.where(
                active,
                distribution.entropy(),
                torch.zeros_like(entropy),
            )
            entropy_steps += active.float()
            hidden, selected, counts, chooses_option = self.policy._advance(
                hidden,
                values,
                selected,
                counts,
                choice,
                active,
            )
            for row, index in enumerate(choice.tolist()):
                if chooses_option[row]:
                    actions[row].append(index)

            if batch.targets is None:
                finished |= (active & ~chooses_option) | (counts >= batch.maximum)
                if finished.all():
                    break

        return PolicyOutput(
            actions,
            log_probability,
            entropy / entropy_steps.clamp_min(1.0),
            state_value,
            torch.stack(outputs, dim=1) if outputs else None,
        )

    @torch.inference_mode()
    def act(self, batch: DecisionBatch, sample: bool = True) -> PolicyOutput:
        """Return sampled or greedy actions with their rollout statistics."""
        return self._sequence(batch, sample)

    def evaluate_actions(self, batch: DecisionBatch) -> PolicyOutput:
        """Rescore labeled actions for a policy-gradient update."""
        if batch.targets is None:
            raise ValueError("action evaluation requires labeled decisions")
        return self._sequence(batch, False)


def load_checkpoint(path: Path) -> dict[str, Any]:
    """Load a model or training checkpoint on CPU."""
    return torch.load(path, map_location="cpu", weights_only=True)


def load_model(
    path: Path,
    device: torch.device | str = "cpu",
) -> EntityPointerPolicy:
    """Load a policy model for inference or fine-tuning."""
    checkpoint = load_checkpoint(path)
    model = EntityPointerPolicy(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval()


def load_actor_critic(
    path: Path,
    device: torch.device | str = "cpu",
) -> ActorCriticPolicy:
    """Load a v1 policy or a published actor-critic model."""
    checkpoint = load_checkpoint(path)
    policy = EntityPointerPolicy(ModelConfig(**checkpoint["model_config"]))
    policy.load_state_dict(checkpoint["model"])
    model = ActorCriticPolicy(policy)
    if "value_head" in checkpoint:
        model.value_head.load_state_dict(checkpoint["value_head"])
    return model.to(device).eval()

