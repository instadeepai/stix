# Copyright 2026 InstaDeep Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Orbax-backed checkpointing for the stix training loop.

A thin wrapper around :class:`orbax.checkpoint.CheckpointManager` that persists
a :class:`~stix.training.training_state.TrainingState` to a directory of choice.

By default, we keep the best-N as well as the latest (in case the run gets cancelled).
"""

import json
import os
from typing import Any, Mapping, TypeAlias

import jax
import orbax.checkpoint as ocp
import pydantic
from etils import epath
from flax import nnx
from orbax.checkpoint.checkpoint_managers import (
    AnyPreservationPolicy,
    BestN,
    LatestN,
)

from stix.training.ema import get_debiased_ema
from stix.training.training_state import TrainingState

PathLike: TypeAlias = str | os.PathLike
DatasetState: TypeAlias = Any | Mapping[str, Any]

_RUN_METADATA_FILENAME = "run_metadata.json"


def _eval_loss_metric(metrics: dict[str, float]) -> float:
    """Ranking key for best-checkpoint selection (lower is better)."""
    return metrics["eval_loss"]


def _run_metadata_path(run_dir: str) -> epath.Path:
    """Location of a run's metadata file: ``<run_dir>/run_metadata.json``."""
    return epath.Path(run_dir) / _RUN_METADATA_FILENAME


def save_run_metadata(run_dir: str, metadata: dict[str, Any]) -> None:
    """Persist run-level metadata as JSON at the run root.

    This is run-level (not checkpoint-level) state, so it lives beside
    ``checkpoints/`` and is read/written without an Orbax ``CheckpointManager``.
    Generic by design: ``metadata`` is any JSON-serialisable dict and the caller
    decides the contents (e.g. a serialised run config plus a human label).

    Args:
        run_dir: The run root (parent of ``checkpoints/``).
        metadata: A JSON-serialisable dict describing the run.
    """
    _run_metadata_path(run_dir).write_text(
        json.dumps(metadata, indent=2, sort_keys=True)
    )


def read_run_metadata(run_dir: str) -> dict[str, Any] | None:
    """Read metadata written by :func:`save_run_metadata`, or ``None`` if absent.

    Manager-free on purpose: this only reads a JSON sibling file, so it never
    constructs a ``CheckpointManager`` and so cannot race a *live* run's
    in-flight checkpoint save (the failure mode that constructing one has).

    Args:
        run_dir: The run root (parent of ``checkpoints/``).

    Returns:
        The metadata dict, or ``None`` if the metadata file does not exist.
    """
    path = _run_metadata_path(run_dir)
    if not path.exists():
        return None
    return json.loads(path.read_text())


class CheckpointerConfig(pydantic.BaseModel):
    """Configuration for :class:`Checkpointer`.

    Attributes:
        checkpoint_dir: Directory to write to.
        max_to_keep: Maximum number of checkpoints to retain. ``None``
            disables pruning entirely, which is the right setting for
            read-only managers pointed at a foreign directory. Default is 5.
        save_interval_steps: Save a checkpoint every N training steps.
            The default is 100 training steps.
        keep_best: Which ``max_to_keep`` to save. If True we keep the best by ``eval_loss``
            (plus the most recent!), else keep the most recent ``max_to_keep``.
        enable_async_checkpointing: Whether Orbax writes asynchronously.
            Defaults to ``False`` (synchronous) in case of crashes.
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    checkpoint_dir: PathLike
    max_to_keep: pydantic.PositiveInt | None = 5
    save_interval_steps: pydantic.PositiveInt = 100
    keep_best: bool = True
    enable_async_checkpointing: bool = False


class Checkpointer:
    """Persist and restore a :class:`~stix.training.training_state.TrainingState` via Orbax."""

    Config = CheckpointerConfig

    def __init__(self, config: CheckpointerConfig) -> None:
        """Build the underlying Orbax ``CheckpointManager``.

        Args:
            config: The checkpointer configuration.
        """
        self.config = config

        # In read-only mode the manager must never mutate the directory: no
        # creation, no tmp cleanup. This is what makes it safe to point at a
        # live run (cleanup would otherwise delete the trainer's in-flight save).
        read_only = config.max_to_keep is None
        mutate = not read_only

        if config.keep_best:
            # Best-N + LatestN(1) for deployment so the most recent checkpoint
            # always survives culling for resume.
            options = ocp.CheckpointManagerOptions(
                save_interval_steps=config.save_interval_steps,
                create=mutate,
                cleanup_tmp_directories=mutate,
                read_only=read_only,
                enable_async_checkpointing=config.enable_async_checkpointing,
                best_fn=_eval_loss_metric,
                best_mode="min",
                preservation_policy=AnyPreservationPolicy(
                    [
                        BestN(
                            get_metric_fn=_eval_loss_metric,
                            reverse=True,
                            n=config.max_to_keep,
                        ),
                        LatestN(n=1),
                    ]
                ),
            )
        else:
            options = ocp.CheckpointManagerOptions(
                save_interval_steps=config.save_interval_steps,
                create=mutate,
                cleanup_tmp_directories=mutate,
                read_only=read_only,
                enable_async_checkpointing=config.enable_async_checkpointing,
                max_to_keep=config.max_to_keep,
            )

        self._manager = ocp.CheckpointManager(config.checkpoint_dir, options=options)

    def save(
        self,
        training_state: TrainingState,
        step: int,
        eval_loss: float,
        dataset_state: DatasetState | None = None,
    ) -> None:
        """Persist ``training_state`` under ``step`` as independently-loadable items.

        The state is split across four Orbax items: ``params``, ``opt_state``,
        ``ema_params`` (the *raw*, non-de-biased EMA params) and ``extra`` (the small scalars),
        so each can be restored on its own.
        In particular, sampling can easily recover the de-biased params via
        ``ema_params`` + ``extra`` (see :meth:`restore_ema`).

        Args:
            training_state: The training state to persist (pulled to host first).
            step: The step number used as the checkpoint key.
            eval_loss: Validation loss at this step, used for best-checkpoint
                ranking and stored in the checkpoint metrics.
            dataset_state: The dataset state to persist.
                If None, the dataset state will not be saved. Defaults to None.
        """
        training_state = jax.device_get(training_state)
        if dataset_state is not None:
            dataset_state = jax.device_get(dataset_state)

        checkpoint_dict = {
            "params": ocp.args.PyTreeSave(training_state.params),
            "opt_state": ocp.args.PyTreeSave(training_state.opt_state),
            "ema_params": ocp.args.PyTreeSave(training_state.ema_params),
            "extra": ocp.args.PyTreeSave(
                {
                    "num_steps": training_state.num_steps,
                    "ema_decay": training_state.ema_decay,
                    "rng_key": training_state.rng_key,
                }
            ),
        }
        if dataset_state is not None:
            checkpoint_dict["dataset_state"] = ocp.args.PyTreeSave(dataset_state)

        save_kwargs = {
            "args": ocp.args.Composite(**checkpoint_dict),
            "metrics": {"eval_loss": float(eval_loss)},
            "custom_metadata": {"num_steps": int(training_state.num_steps)},
        }

        self._manager.save(step, **save_kwargs)

    def restore(
        self,
        training_state: TrainingState,
        dataset_state: DatasetState | None = None,
        step: int | None = None,
        restore_optimizer_state: bool = True,
        restore_rng_key: bool = True,
    ) -> tuple[TrainingState, DatasetState | None] | None:
        """Restore a :class:`~stix.training.training_state.TrainingState` and ``DatasetState``, defaulting to the latest checkpoint.

        If the checkpoint does not contain a dataset state, the returned dataset state will be None.

        Args:
            training_state: A freshly-initialised training state of the right structure; its
                static ``graphdef`` and (when a restore flag is off) its
                ``opt_state`` / ``rng_key`` are reused.
            dataset_state: A freshly-initialised dataset state of the right structure.
                If None or if the checkpoint does not contain a dataset state,
                the returned dataset state will be None. Defaults to None.
            step: Step to restore. If None, defaults to ``latest_step()``.
                Defaults to ``None``.
            restore_optimizer_state: Keep the checkpointed optimizer state
                (Adam momentum). When ``False``, reset to ``training_state``'s.
            restore_rng_key: Continue the checkpointed PRNG stream. When
                ``False``, reset to ``training_state``'s key.

        Returns:
            The restored training state and dataset state, or ``None`` if no checkpoint exists.
            If no dataset state is provided or if the checkpoint does not contain a dataset state,
            the returned dataset state will be None.

        Raises:
            ValueError: If the restored ``num_steps`` disagrees with the
                ``num_steps`` recorded in the checkpoint's custom metadata.
        """
        step = self._resolve_step(step)
        # No checkpoint to restore: signal the caller to start from scratch.
        if step is None:
            return None

        # Read only the items we need.
        items = {
            "params": ocp.args.PyTreeRestore(training_state.params),
            "ema_params": ocp.args.PyTreeRestore(training_state.ema_params),
            "extra": ocp.args.PyTreeRestore(),
        }
        if restore_optimizer_state:
            items["opt_state"] = ocp.args.PyTreeRestore(training_state.opt_state)
        if dataset_state is not None:
            items["dataset_state"] = ocp.args.PyTreeRestore(dataset_state)
        restored = self._manager.restore(step, args=ocp.args.Composite(**items))

        extra = restored["extra"]
        meta = self._manager.metadata(step)
        meta_num_steps = 0
        if meta is not None and meta.custom_metadata is not None:
            meta_num_steps = meta.custom_metadata.get("num_steps", 0)
        if meta_num_steps > 0 and int(extra["num_steps"]) != meta_num_steps:
            raise ValueError(
                f"Checkpoint restore mismatch: metadata num_steps={meta_num_steps} "
                f"but restored num_steps={int(extra['num_steps'])}. This usually "
                f"means the template pytree structure does not match the checkpoint."
            )

        restored_dataset_state = (
            restored["dataset_state"] if dataset_state is not None else None
        )

        restored_training_state = training_state.replace(
            params=restored["params"],
            opt_state=restored["opt_state"]
            if restore_optimizer_state
            else training_state.opt_state,
            ema_params=restored["ema_params"],
            num_steps=restored["extra"]["num_steps"],
            ema_decay=extra["ema_decay"],
            rng_key=extra["rng_key"] if restore_rng_key else training_state.rng_key,
        )

        return (restored_training_state, restored_dataset_state)

    def restore_ema(
        self,
        template: nnx.State,
        step: int | None = None,
    ) -> nnx.State:
        """Restore the debiased EMA parameters for offline sampling/evaluation.

        Reads only the ``ema_params`` and ``extra`` items, never ``params`` or
        ``opt_state``, then applies the same bias correction the live loop uses
        for evaluation (``ema / (1 - decay ** num_steps)``).

        Args:
            template: A params :class:`~flax.nnx.State` of the right structure,
                e.g. ``nnx.split(model, nnx.Param)[1]``. No optimizer or full
                training state is needed.
            step: Step to restore. Defaults to ``latest_step()``.

        Returns:
            The debiased EMA parameters, ready to merge with a ``graphdef``.

        Raises:
            ValueError: If no checkpoint exists.
        """
        step = self._resolve_step(step)
        if step is None:
            raise ValueError(
                f"No checkpoint to restore in {self.config.checkpoint_dir}."
            )
        restored = self._manager.restore(
            step,
            args=ocp.args.Composite(
                ema_params=ocp.args.PyTreeRestore(template),
                extra=ocp.args.PyTreeRestore(),
            ),
        )
        ema_params = restored["ema_params"]
        num_steps = int(restored["extra"]["num_steps"])
        if num_steps == 0:
            # No updates yet: EMA is still its zero init, nothing to debias.
            return ema_params
        debiased = get_debiased_ema(
            ema_params, float(restored["extra"]["ema_decay"]), num_steps
        )
        # Restored leaves are host (numpy) arrays, so numpy type promotion can
        # widen the correction to float64; cast back to each leaf's stored dtype
        # so offline sampling gets params matching the (float32) model.
        return jax.tree.map(
            lambda original, d: d.astype(original.dtype), ema_params, debiased
        )

    def _resolve_step(self, step: int | None) -> int | None:
        """Return ``step``, or the latest step when ``None``.

        Return ``None`` if no checkpoint exists.
        """
        if step is None:
            step = self._manager.latest_step()
        return step

    def latest_step(self) -> int | None:
        """Return the most recent checkpoint step, or ``None``."""
        return self._manager.latest_step()

    def best_step(self) -> int | None:
        """Return the best-``eval_loss`` checkpoint step, or ``None``."""
        return self._manager.best_step()

    def wait_until_finished(self) -> None:
        """Block until any pending asynchronous saves complete."""
        self._manager.wait_until_finished()
