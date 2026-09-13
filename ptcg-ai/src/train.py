from __future__ import annotations

import json
import random
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

import torch

from base_data import DecisionBatch, file_sha256
from common import EntityPointerPolicy, ModelConfig, load_checkpoint


BatchSource = Callable[[str, int, int, bool], Iterator[DecisionBatch]]


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 32
    accumulate: int = 4
    epochs: int = 3
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    gradient_clip: float = 1.0
    amp: str = "auto"
    evaluate_every: int = 2000
    validation_batches: int = 200
    log_every: int = 50
    max_steps: int = 0
    seed: int = 42


def set_seed(seed: int) -> None:
    """Seed Python and PyTorch."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def amp_mode(device: torch.device, requested: str) -> str:
    """Resolve the requested mixed-precision mode for a device."""
    if requested == "auto":
        if device.type != "cuda":
            return "none"
        return "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    if requested not in {"bf16", "fp16", "none"}:
        raise ValueError("amp must be auto, bf16, fp16, or none")
    if requested != "none" and device.type != "cuda":
        raise ValueError("mixed precision requires CUDA")
    if requested == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("this GPU does not support BF16")
    return requested


def amp_context(device: torch.device, mode: str):
    """Create the selected mixed-precision context."""
    if mode == "none":
        return nullcontext()
    dtype = torch.bfloat16 if mode == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype)


@torch.inference_mode()
def evaluate(
    model: EntityPointerPolicy,
    batches: BatchSource,
    batch_size: int,
    maximum_batches: int,
    device: torch.device,
    amp: str,
    seed: int,
) -> float:
    """Return the target-weighted validation loss."""
    model.eval()
    total_loss = 0.0
    total_targets = 0
    for index, batch in enumerate(batches("validation", batch_size, seed, True)):
        if maximum_batches and index >= maximum_batches:
            break
        batch = batch.to(device)
        with amp_context(device, amp):
            loss = model.loss(batch)
        target_count = int((batch.targets != -100).sum().item())
        total_loss += loss.item() * target_count
        total_targets += target_count
    if not total_targets:
        raise ValueError("validation data produced no targets")
    return total_loss / total_targets


def optimizer_step(
    model: EntityPointerPolicy,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    gradient_clip: float,
    gradient_multiplier: float = 1.0,
) -> None:
    """Apply one clipped optimizer step."""
    scaler.unscale_(optimizer)
    if gradient_multiplier != 1.0:
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(gradient_multiplier)
    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)


def save_checkpoint(path: Path, state: dict[str, Any]) -> None:
    """Atomically save one training checkpoint."""
    temporary = path.with_name(f"{path.name}.tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def write_json(path: Path, value: dict[str, Any]) -> None:
    """Atomically write readable JSON."""
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def print_event(event: str, **values: Any) -> None:
    """Print one compact training event."""
    print(
        json.dumps({"event": event, **values}, ensure_ascii=False, sort_keys=True),
        flush=True,
    )

def train(
    name: str,
    batches: BatchSource,
    dataset: dict[str, Any],
    output_dir: Path,
    work_dir: Path,
    config: TrainingConfig | None = None,
    model_config: ModelConfig | None = None,
    initial_model: Path | None = None,
    resume: bool = False,
    device: str | torch.device | None = None,
    info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train and publish one base or expert model."""
    settings = config or TrainingConfig()
    if min(
        settings.batch_size,
        settings.accumulate,
        settings.epochs,
        settings.log_every,
    ) <= 0:
        raise ValueError("batch size, accumulation, epochs, and log interval must be positive")
    if settings.learning_rate <= 0 or settings.weight_decay < 0 or settings.gradient_clip <= 0:
        raise ValueError("invalid optimizer settings")
    if settings.evaluate_every < 0 or settings.validation_batches < 0 or settings.max_steps < 0:
        raise ValueError("evaluation and step limits cannot be negative")
    if resume and initial_model:
        raise ValueError("resume and initial_model cannot be used together")

    selected_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    selected_amp = amp_mode(selected_device, settings.amp)
    dataset_sha256 = dataset["dataset_sha256"]

    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.pt"
    record_path = output_dir / "record.json"
    last_path = work_dir / "last.pt"
    best_path = work_dir / "best.pt"
    if not resume and any(path.exists() for path in (model_path, record_path, last_path, best_path)):
        raise FileExistsError(f"training output already exists: {name}")
    if resume and not last_path.exists():
        raise FileNotFoundError(f"no checkpoint to resume: {last_path}")

    set_seed(settings.seed)
    torch.set_float32_matmul_precision("high")
    resume_checkpoint = load_checkpoint(last_path) if resume else None
    initial_checkpoint = load_checkpoint(initial_model) if initial_model else None
    source_checkpoint = resume_checkpoint or initial_checkpoint
    architecture = (
        ModelConfig(**source_checkpoint["model_config"])
        if source_checkpoint
        else model_config or ModelConfig()
    )
    model = EntityPointerPolicy(architecture).to(selected_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=selected_amp == "fp16")

    epoch = 0
    resume_batch = 0
    step = 0
    best_loss = float("inf")
    initialized_from = None
    if resume_checkpoint:
        if resume_checkpoint["dataset_sha256"] != dataset_sha256:
            raise ValueError("checkpoint belongs to a different dataset")
        if resume_checkpoint["training_config"] != asdict(settings):
            raise ValueError("training settings differ from the checkpoint")
        if resume_checkpoint["amp"] != selected_amp:
            raise ValueError("mixed precision differs from the checkpoint")
        model.load_state_dict(resume_checkpoint["model"])
        optimizer.load_state_dict(resume_checkpoint["optimizer"])
        scaler.load_state_dict(resume_checkpoint["scaler"])
        epoch = int(resume_checkpoint["epoch"])
        resume_batch = int(resume_checkpoint["next_batch"])
        step = int(resume_checkpoint["step"])
        best_loss = float(resume_checkpoint["best_validation_loss"])
        initialized_from = resume_checkpoint["initialized_from"]
    elif initial_checkpoint:
        model.load_state_dict(initial_checkpoint["model"])
        initialized_from = {
            "path": initial_model.as_posix(),
            "sha256": file_sha256(initial_model),
            "dataset_sha256": initial_checkpoint.get("dataset_sha256"),
        }

    print_event(
        "start",
        name=name,
        device=torch.cuda.get_device_name(selected_device) if selected_device.type == "cuda" else "cpu",
        amp=selected_amp,
        parameters=sum(parameter.numel() for parameter in model.parameters()),
        step=step,
    )

    def checkpoint_state(current_epoch: int, next_batch: int) -> dict[str, Any]:
        """Build the current resumable training state."""
        return {
            "model": model.state_dict(),
            "model_config": asdict(model.config),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "training_config": asdict(settings),
            "dataset_sha256": dataset_sha256,
            "epoch": current_epoch,
            "next_batch": next_batch,
            "step": step,
            "best_validation_loss": best_loss,
            "amp": selected_amp,
            "initialized_from": initialized_from,
        }

    last_evaluation_step = -1

    def validate(current_epoch: int, next_batch: int) -> None:
        """Evaluate and update the best and latest checkpoints."""
        nonlocal best_loss, last_evaluation_step
        validation_loss = evaluate(
            model,
            batches,
            settings.batch_size,
            settings.validation_batches,
            selected_device,
            selected_amp,
            settings.seed,
        )
        improved = validation_loss < best_loss or not best_path.exists()
        if improved:
            best_loss = validation_loss
        state = checkpoint_state(current_epoch, next_batch)
        if improved:
            save_checkpoint(best_path, state)
        save_checkpoint(last_path, state)
        last_evaluation_step = step
        print_event(
            "validation",
            epoch=current_epoch,
            step=step,
            loss=validation_loss,
            best=best_loss,
        )
        model.train()

    running_loss = 0.0
    running_examples = 0
    window_started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    stop = bool(settings.max_steps and step >= settings.max_steps)

    for current_epoch in range(epoch, settings.epochs):
        if stop:
            break
        model.train()
        accumulated = 0
        next_batch = 0
        for batch_index, batch in enumerate(
            batches("train", settings.batch_size, settings.seed + current_epoch, True)
        ):
            next_batch = batch_index + 1
            if current_epoch == epoch and batch_index < resume_batch:
                continue
            batch = batch.to(selected_device)
            with amp_context(selected_device, selected_amp):
                loss = model.loss(batch)
            scaler.scale(loss / settings.accumulate).backward()
            accumulated += 1
            examples = len(batch.minimum)
            running_loss += loss.detach().item() * examples
            running_examples += examples
            if accumulated < settings.accumulate:
                continue

            optimizer_step(model, optimizer, scaler, settings.gradient_clip)
            accumulated = 0
            step += 1
            if step % settings.log_every == 0:
                elapsed = max(time.perf_counter() - window_started, 1e-6)
                print_event(
                    "train",
                    epoch=current_epoch,
                    step=step,
                    loss=running_loss / running_examples,
                    decisions_per_second=running_examples / elapsed,
                )
                running_loss = 0.0
                running_examples = 0
                window_started = time.perf_counter()
            if settings.evaluate_every and step % settings.evaluate_every == 0:
                validate(current_epoch, next_batch)
            if settings.max_steps and step >= settings.max_steps:
                if last_evaluation_step != step:
                    validate(current_epoch, next_batch)
                stop = True
                break

        if stop:
            break
        if accumulated:
            optimizer_step(
                model,
                optimizer,
                scaler,
                settings.gradient_clip,
                settings.accumulate / accumulated,
            )
            step += 1
        resume_batch = 0
        validate(current_epoch + 1, 0)
        if settings.max_steps and step >= settings.max_steps:
            break

    if not best_path.exists():
        validate(epoch, resume_batch)

    best = load_checkpoint(best_path)
    published = {
        "model": best["model"],
        "model_config": best["model_config"],
    }
    temporary_model = model_path.with_name(f"{model_path.name}.tmp")
    torch.save(published, temporary_model)
    temporary_model.replace(model_path)
    record = {
        "name": name,
        **(info or {}),
        "model": {
            "path": model_path.name,
            "sha256": file_sha256(model_path),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "config": best["model_config"],
        },
        "dataset": dataset,
        "training": {
            "config": asdict(settings),
            "device": torch.cuda.get_device_name(selected_device) if selected_device.type == "cuda" else "cpu",
            "amp": selected_amp,
            "steps": step,
            "best_step": best["step"],
            "best_validation_loss": best_loss,
            "initialized_from": initialized_from,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        },
    }
    write_json(record_path, record)
    print_event("complete", name=name, step=step, best_validation_loss=best_loss)
    return record
