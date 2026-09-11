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

"""Training IO handler for the stix training loop."""

import logging
import os
import shutil
import time
from concurrent.futures import Future
from enum import Enum
from pathlib import Path
from typing import Any, Callable, TypeAlias

import jax
import pydantic
from jax.experimental import multihost_utils
from typing_extensions import Annotated

from stix.training.checkpointer import (
    Checkpointer,
    CheckpointerConfig,
    DatasetState,
    PathLike,
)
from stix.training.training_state import TrainingState

Source: TypeAlias = PathLike

PositiveInt = Annotated[int, pydantic.Field(gt=0)]

logger = logging.getLogger("stix")


class CheckpointRestorationError(Exception):
    """Exception to be raised if issues occur during checkpoint restoration."""


class TrainingIOHandlerConfig(pydantic.BaseModel):
    """Pydantic config holding all settings relevant for the training IO handler.

    Attributes:
        checkpoint_dir: Root checkpoint directory, e.g. a local path or any
            URI scheme Orbax supports. When `None`,
            the `clear_previous_checkpoints` guard is skipped.
            Defaults to `None`.
        restore_dir: Directory to restore checkpoints from. If `None`, will
            default to `checkpoint_dir`.
        max_to_keep: Maximum number of checkpoints to retain.
            The default is 5.
        save_interval_steps: Save a checkpoint every N training steps.
            The default is `None`, which disables checkpointing.
        keep_best: Which `max_to_keep` to save. If True we keep the best by
            `eval_loss` (plus the most recent!), else keep the most recent
            `max_to_keep`. The default is `True`.
        restore_checkpoint_if_exists: Whether to restore a previous checkpoint if it
            exists. By default, this is `False`.
        step_to_restore: The step number to restore. The default is `None`, which
            means the latest step will be restored.
        restore_optimizer_state: Whether to also restore the optimizer state.
            Default is `True`.
        clear_previous_checkpoints: Whether to clear the previous checkpoints if
            any exist. Note that this setting can not be set to
            `True` if one selects to restore a checkpoint.
            The default is `False`.
        enable_async_checkpointing: Whether Orbax should write checkpoints
            asynchronously. Defaults to `False`.
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    checkpoint_dir: PathLike | None = None
    restore_dir: PathLike | None = None

    # Checkpointer settings
    max_to_keep: PositiveInt = 5
    save_interval_steps: PositiveInt | None = None
    keep_best: bool = True
    enable_async_checkpointing: bool = False

    # Restoration settings
    restore_checkpoint_if_exists: bool = False
    step_to_restore: PositiveInt | None = None
    restore_optimizer_state: bool = True
    clear_previous_checkpoints: bool = False

    @pydantic.model_validator(mode="after")
    def _disallow_clear_with_restore(self) -> "TrainingIOHandlerConfig":
        if self.clear_previous_checkpoints and self.restore_checkpoint_if_exists:
            raise ValueError(
                "clear_previous_checkpoints=True is incompatible with "
                "restore_checkpoint_if_exists=True: clearing first would "
                "delete the checkpoints intended for restoration."
            )
        return self


class LoggingCategory(Enum):
    """Enum class for logging categories.

    These values provide a signal to a logging function what type of data is
    being logged.

    Attributes:
        BEST_MODEL: Information about the current best model is logged.
        TRAIN_METRICS: Metrics for the training set are logged.
        EVAL_METRICS: Metrics for the validation set are logged.
        TEST_METRICS: Metrics for the test set are logged.
        SYSTEM_METRICS: Per-process system metrics (runtime, throughput) are logged.
        CLEANUP_AFTER_CKPT_RESTORATION: Allows the logger to clean itself up after a
            checkpoint has been restored.
    """

    BEST_MODEL = 0
    TRAIN_METRICS = 1
    EVAL_METRICS = 2
    TEST_METRICS = 3
    SYSTEM_METRICS = 4
    CLEANUP_AFTER_CKPT_RESTORATION = 5


LoggerFn: TypeAlias = Callable[[LoggingCategory, dict[str, Any], int], None]


class TrainingIOHandler:
    """An IO handler class for the training loop.

    This handles checkpointing as well as specialised logging, e.g., to some external
    logger that a user can provide. Checkpointing is delegated to
    :class:`stix.training.checkpointer.Checkpointer` instances for
    training state and potentially dataset state saving and restoration.
    """

    Config = TrainingIOHandlerConfig

    def __init__(
        self,
        config: TrainingIOHandlerConfig | None = None,
        data_upload_fn: Callable[[Source], Future | None] | None = None,
    ) -> None:
        """Initialise the training IO handler and its checkpointers.

        Args:
            config: The training IO handler pydantic config. Can be `None` in which
                case the default config will be used. Default is `None`.
            data_upload_fn: A data upload function to a remote storage.
                This is optional, and set to None as default.
                This function should just take in a source path, and then
                the upload location can be user-defined within that
                function. The function can be asynchronous in which case it
                should return a Future.
        """
        if config is None:
            self.config = TrainingIOHandlerConfig()
        else:
            self.config = config

        self._checkpoint_dir: os.PathLike | None = None
        if self.config.checkpoint_dir is not None:
            cd = self.config.checkpoint_dir
            self._checkpoint_dir = (
                cd if isinstance(cd, os.PathLike) else Path(cd).resolve()
            )

        self._data_upload_fn = data_upload_fn
        self._future: Future | None = None

        if self.config.clear_previous_checkpoints:
            self._clear_checkpoints()

        self._checkpointer = self._create_checkpointer()

        rd = self.config.restore_dir
        if rd is not None:
            rd = rd if isinstance(rd, os.PathLike) else Path(rd).resolve()
        self.restore_dir = rd if rd is not None else self._checkpoint_dir

        self._restore_checkpointer = None
        if self.restore_dir is not None:
            if self.restore_dir == self._checkpoint_dir:
                self._restore_checkpointer = self._checkpointer
            else:
                self._restore_checkpointer = self._create_restore_checkpointer()

        self.loggers: list[LoggerFn] = []

    def _create_checkpointer(self) -> Checkpointer | None:
        """Create the training checkpointer from the IO handler config."""
        if (
            self._checkpoint_dir is None
            or self.config.save_interval_steps is None
            or self.config.max_to_keep is None
        ):
            return None
        config = CheckpointerConfig(
            checkpoint_dir=self._checkpoint_dir,
            max_to_keep=self.config.max_to_keep,
            save_interval_steps=self.config.save_interval_steps,
            keep_best=self.config.keep_best,
            enable_async_checkpointing=self.config.enable_async_checkpointing,
        )
        return Checkpointer(config)

    def _create_restore_checkpointer(self) -> Checkpointer | None:
        """Create a read-only checkpointer pointed at the restore directory."""
        if self.restore_dir is None:
            return None
        config = CheckpointerConfig(
            checkpoint_dir=self.restore_dir,
            max_to_keep=None,
            enable_async_checkpointing=self.config.enable_async_checkpointing,
        )
        return Checkpointer(config)

    def attach_logger(self, logger: LoggerFn) -> None:
        """Attach one training loop logging function to the IO handler.

        The logging function must take in three parameters and should not return
        anything. The three parameters are a logging category which describes what
        type of data is logged (it is an enum), the data dictionary to log, and
        the current step number.

        Args:
            logger: The logging function to add.
        """
        self.loggers.append(logger)

    def log(
        self, category: LoggingCategory, to_log: dict[str, Any], step_number: int
    ) -> None:
        """Log data via the logging functions stored in this class.

        Args:
            category: A logging category which describes what type of data is
                logged (it is an enum).
            to_log: A data dictionary to log (typically, metrics).
            step_number: The current step number.
        """
        for logger in self.loggers:
            logger(category, to_log, step_number)

    def _run_upload(self) -> None:
        """Upload the checkpoint directory using the configured upload function.

        Waits for any previously returned :class:`Future` before starting the
        next upload, ensuring sequential upload ordering.
        """
        if self._data_upload_fn is None or self._checkpoint_dir is None:
            return

        self.wait_until_finished()
        self._future = self._data_upload_fn(self._checkpoint_dir)

    def save_checkpoint(
        self,
        training_state: TrainingState,
        step_number: int,
        eval_loss: float,
        dataset_state: DatasetState | None = None,
    ) -> None:
        """Save a model checkpoint and upload it if an upload function is configured.

        Args:
            training_state: The training state to save.
            step_number: The current step number.
            eval_loss: Scalar evaluation loss used for best-checkpoint ranking.
            dataset_state: The dataset state to save.
                If None, the dataset state will not be saved. Defaults to None.
        """
        if self._checkpointer is None:
            return

        logger.info("Saving checkpoint at step %s...", step_number)
        self._checkpointer.save(
            training_state,
            step=step_number,
            eval_loss=eval_loss,
            dataset_state=dataset_state,
        )

        self._run_upload()

    def restore_checkpoint(
        self,
        training_state: TrainingState,
        dataset_state: DatasetState | None = None,
        restore_optimizer_state: bool = True,
        restore_rng_key: bool = True,
    ) -> tuple[TrainingState, DatasetState | None]:
        """Restore a :class:`~stix.training.training_state.TrainingState`, defaulting to the latest checkpoint.

        Args:
            training_state: A freshly-initialised training state
                of the right structure; its static ``graphdef``
                and (when a restore flag is off) its
                ``opt_state`` / ``rng_key`` are reused.
            dataset_state: A freshly-initialised dataset state of the right structure.
                If None or if the checkpoint does not contain a dataset state,
                the returned dataset state will be None. Defaults to None.
            restore_optimizer_state: Keep the checkpointed optimizer state
                (Adam momentum). When ``False``, reset to ``training_state``'s.
            restore_rng_key: Continue the checkpointed PRNG stream. When
                ``False``, reset to ``training_state``'s key.

        Returns:
            The restored training state and dataset state, or the inputs unchanged
            if no checkpoint exists.

        Raises:
            CheckpointRestorationError: If restoration is requested but
                checkpointing is disabled.
        """
        if not self.config.restore_checkpoint_if_exists:
            return (training_state, dataset_state)

        if self._restore_checkpointer is None:
            raise CheckpointRestorationError(
                "Cannot restore training state as checkpointing is disabled."
            )

        start_time = time.perf_counter()

        restored_states = self._restore_checkpointer.restore(
            training_state,
            dataset_state=dataset_state,
            step=self.config.step_to_restore,
            restore_optimizer_state=restore_optimizer_state,
            restore_rng_key=restore_rng_key,
        )

        if restored_states is None:
            logger.info("No checkpoint found to restore — starting from scratch.")
            return (training_state, dataset_state)

        restored_training_state, restored_dataset_state = restored_states

        logger.debug(
            "Checkpoint was restored in %.2f sec.",
            time.perf_counter() - start_time,
        )

        return (restored_training_state, restored_dataset_state)

    def _clear_checkpoints(self) -> None:
        """Remove all existing checkpoints from the checkpoint directory.

        Called during `__init__` when `clear_previous_checkpoints` is `True`.
        """
        if self._checkpoint_dir is None:
            return
        if jax.process_index() == 0:
            logger.info("Clearing previous checkpoints in %s.", self._checkpoint_dir)
            ckpt_dir = self._checkpoint_dir
            if Path(ckpt_dir).exists():
                shutil.rmtree(ckpt_dir)
                logger.debug("Removed %s.", ckpt_dir)

        if jax.process_count() > 1:
            multihost_utils.sync_global_devices("clear_checkpoints")

    def wait_until_finished(self) -> None:
        """Wait until local checkpoints and uploads are finished due to their asynchronous nature.

        To be called at the end of a training run.
        """
        if self._checkpointer is not None:
            self._checkpointer.wait_until_finished()
        if self._future is not None:
            self._future.result()
