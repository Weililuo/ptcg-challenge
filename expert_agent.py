"""Torch expert opponents for the DAgger rollout pool.

This is the ONLY module in the pipeline that touches torch. It is imported
lazily from inside ``rollout_worker.run_single_match`` / ``arena_judge.play_game``
so that pure-XGB stages (retrain, mirror-only matches) never pay the torch
import cost.

The experts under ``ptcg_owen/models/experts`` are ``EntityPointerPolicy``
checkpoints. ``ptcg_owen/run.py`` already knows how to turn one into a plain
``(observation, configuration) -> list[int]`` callable, so we reuse it rather
than reimplementing ``encode_decision``/``collate_decisions`` here.

Locking contract (read before editing)
--------------------------------------
Nothing here may acquire a lock while already holding one. The previous version
nested ``_load_runtime()`` inside ``_load_policy()``'s ``with _runtime_lock``
block; ``threading.Lock`` is not reentrant, so the second acquire blocked the
worker forever on its first torch-expert match, at 0% CPU and with no
traceback. The two locks below are therefore strictly independent, and every
slow operation (``import torch``, ``spec.loader.exec_module``, ``load_model``)
happens OUTSIDE any lock. Redundant concurrent loads are possible and harmless
— ``dict.setdefault`` keeps exactly one winner.
"""

from __future__ import annotations

import importlib.util
import threading
from pathlib import Path
from typing import Any

import dagger_common as dc

_runtime: Any = None
_runtime_lock = threading.Lock()

# One loaded policy per (worker process, model path). Loading is ~14MB and
# ~50ms; the pool runs thousands of matches per worker, so this matters.
_POLICY_CACHE: dict[str, Any] = {}
_cache_lock = threading.Lock()


def _build_runtime() -> Any:
    """Import ``ptcg_owen/run.py`` by absolute path.

    It cannot be a normal ``import run`` because the project root already has
    ``run_match.py`` / ``run_dagger_loop.py`` and we must not depend on import
    order. ``run.py`` inserts its own ``src/`` into ``sys.path`` on import,
    which is what lets its ``from base_data import ...`` resolve.

    Never call this while holding a lock: ``exec_module`` pulls in torch and
    runs for seconds.
    """
    run_py = dc.OWEN_ROOT / "run.py"
    if not run_py.exists():
        raise FileNotFoundError(f"ptcg_owen run.py not found at {run_py} (check --owen-root)")
    spec = importlib.util.spec_from_file_location("owen_run", run_py)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not build import spec for {run_py}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_runtime() -> Any:
    """Return the loaded ``owen_run`` module, importing it once per process.

    Lock scope covers only the global assignment, so the module import happens
    outside the critical section. Under multiprocessing ``spawn`` each worker
    gets its own module instance and its own ``_runtime``; the lock only guards
    against threads inside one process.
    """
    global _runtime
    if _runtime is not None:
        return _runtime

    module = _build_runtime()

    with _runtime_lock:
        if _runtime is None:
            _runtime = module
        return _runtime


def _load_policy(model_path: str) -> Any:
    """Load a published expert checkpoint, cached per worker process.

    Deliberately lock-free on the hot path, and never holds a lock while
    calling ``_load_runtime`` / ``load_model`` — that nesting is what used to
    deadlock the rollout pool.
    """
    cached = _POLICY_CACHE.get(model_path)
    if cached is not None:
        return cached

    import torch

    # 28 workers on 32 cpus: without this each forward pass fans out to
    # every core and the workers thrash. slurm submits also set
    # OMP/MKL/OPENBLAS_NUM_THREADS=1, but ad-hoc login-node runs do not.
    try:
        torch.set_num_threads(1)
    except RuntimeError:
        pass

    runtime = _load_runtime()
    policy = runtime.load_model(Path(model_path), torch.device("cpu"))

    with _cache_lock:
        return _POLICY_CACHE.setdefault(model_path, policy)


class ExpertOpponentAgent:
    """Adapter that gives a torch expert the ``ai.py`` AIAgent surface.

    ``TrajectoryRecorder`` calls ``observe`` on every observation, reads
    ``player_index``, calls ``finish`` after the match, and drives the agent via
    ``__call__``. The raw torch callable only provides ``__call__``, so the
    other three are supplied here instead of teaching the recorder to probe.
    """

    def __init__(self, model_path: str, deck: list[int], player_index: int) -> None:
        self.player_index = player_index
        self.model_path = model_path
        runtime = _load_runtime()
        import torch

        self._agent = runtime.make_agent(_load_policy(model_path), deck, torch.device("cpu"))

    def observe(self, _observation: Any) -> None:
        """No-op: the transformer is stateless per observation."""
        return None

    def finish(self, _won: bool) -> tuple[float, float]:
        """Return a neutral score pair.

        ``run_single_match`` calls ``finish`` on BOTH sides unconditionally, so
        this must exist and must return a tuple even though the expert's
        trajectory is never recorded into our dataset.
        """
        return (0.0, 0.0)

    def __call__(self, observation: Any, configuration: Any = None) -> list[int]:
        return self._agent(observation, configuration)


def policy_cache_size() -> int:
    """Number of policies resident in this worker (diagnostics)."""
    return len(_POLICY_CACHE)


def warm_policies(model_paths: list[str]) -> int:
    """Preload several experts, e.g. from a worker initializer."""
    for path in model_paths:
        _load_policy(path)
    return len(_POLICY_CACHE)


__all__ = ["ExpertOpponentAgent", "policy_cache_size", "warm_policies"]
