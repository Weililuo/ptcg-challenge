from __future__ import annotations

import json
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from common import load_model
from train import BatchSource, amp_context, amp_mode, print_event, write_json


@torch.inference_mode()
def evaluate_model(
    model_dir: Path,
    batches: BatchSource,
    batch_size: int = 64,
    maximum_batches: int = 0,
    log_every: int = 500,
    amp: str = "auto",
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Evaluate a published model on held-out test data and update its record."""
    if batch_size <= 0 or maximum_batches < 0 or log_every < 0:
        raise ValueError("invalid test settings")

    selected_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    selected_amp = amp_mode(selected_device, amp)
    torch.set_float32_matmul_precision("high")

    model_path = model_dir / "model.pt"
    record_path = model_dir / "record.json"
    model = load_model(model_path, selected_device)
    source = batches("test", batch_size, 0, False)
    if maximum_batches:
        source = islice(source, maximum_batches)

    total_loss = 0.0
    total_targets = 0
    correct_targets = 0
    correct_decisions = 0
    decisions = 0
    batch_count = 0

    for batch_count, batch in enumerate(source, 1):
        batch = batch.to(selected_device)
        with amp_context(selected_device, selected_amp):
            logits = model.teacher_forced_logits(batch)
            loss = F.cross_entropy(
                logits.flatten(0, 1),
                batch.targets.flatten(),
                ignore_index=-100,
                reduction="sum",
            )

        targets = batch.targets
        valid = targets != -100
        predictions = logits.argmax(dim=-1)
        target_count = int(valid.sum().item())
        total_loss += loss.item()
        total_targets += target_count
        correct_targets += int(((predictions == targets) & valid).sum().item())
        correct_decisions += int(
            ((predictions == targets) | ~valid).all(dim=1).sum().item()
        )
        decisions += len(targets)

        if log_every and batch_count % log_every == 0:
            print_event("test", batches=batch_count, decisions=decisions)

    if not total_targets:
        raise ValueError("test data produced no targets")

    result = {
        "method": "teacher_forced",
        "loss": total_loss / total_targets,
        "selection_accuracy": correct_targets / total_targets,
        "decision_accuracy": correct_decisions / decisions,
        "decisions": decisions,
        "selections": total_targets,
        "batches": batch_count,
        "batch_limit": maximum_batches or None,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["test"] = result
    write_json(record_path, record)
    print_event("test_complete", **result)
    return result
