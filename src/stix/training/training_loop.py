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

"""Training loop for stochastic interpolant models.

Step-based training loop: termination is controlled by ``num_steps`` (total
optimizer updates), with periodic evaluation every ``eval_every_n_steps``.

The training and evaluation hot paths are single JIT'd functions built via
``_make_train_step`` / ``_make_eval_step`` factories.
"""

import logging
import time
from typing import Callable, Iterator, Sized

import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from flax import nnx
from jaxtyping import PyTree

from stix.core.gen_model.gen_model import GenerativeModel
from stix.training.ema import get_debiased_ema
from stix.training.evaluation_step import _make_eval_step
from stix.training.loss_pipeline import LossPipeline
from stix.training.training_io_handler import LoggingCategory, TrainingIOHandler
from stix.training.training_loggers import log_metrics_to_line
from stix.training.training_loop_config import TrainingLoopConfig
from stix.training.training_state import TrainingState
from stix.training.training_step import _make_train_step
from stix.typing import PRNGKeyArray, Scalar
from stix.typing.data import Batch

logger = logging.getLogger("stix")


class TrainingLoop:
    """Step-based training loop for stochastic interpolant models.

    The loop runs for ``config.num_steps`` optimizer updates. Evaluation is
    triggered every ``config.eval_every_n_steps`` (if set) and always at the
    end of training. There is no concept of an epoch: ``train_data`` and
    ``val_data`` must be iterators (objects supporting ``next(...)``) that
    yield enough batches for the full run; the caller is responsible for
    any cycling/shuffling.

    Attributes:
        training_state: The current training state.
        best_eval_loss: Best eval loss since this process started; reset on
            resume, not restored (the durable best lives in the checkpoint store).
        best_eval_step: The step at which ``best_eval_loss`` was found (same
            resume caveat).
    """

    Config = TrainingLoopConfig

    def __init__(
        self,
        train_data: Iterator[Batch],
        val_data: Iterator[Batch] | None,
        loss_pipeline: LossPipeline,
        gen_model: GenerativeModel,
        optimizer_tx: optax.GradientTransformation,
        config: TrainingLoopConfig,
        *,
        num_batches_per_eval_step: int | None = 5,
        eval_fn: Callable[[GenerativeModel], float] | None = None,
        io_handler: TrainingIOHandler | None = None,
    ) -> None:
        """Initialise the training loop.

        Args:
            train_data: An iterator yielding training batches. Must support
                ``next(...)`` and yield enough batches for the full run; any
                cycling/shuffling is the caller's responsibility.
            val_data: An optional iterator yielding validation batches.
                Same contract as ``train_data``.
            loss_pipeline: The :class:`~stix.training.loss_pipeline.LossPipeline` object
                (contains coupling, time sampler, batch size config).
            gen_model: The live generative model (``nnx.Module``) to train.
            optimizer_tx: An optax gradient transformation.
            config: The training loop configuration.
            num_batches_per_eval_step: Number of batches consumed per evaluation pass.
                Defaults to 5. Pass ``None`` to instead infer it from
                ``len(val_data)``; this requires ``val_data`` to be sized (or
                ``None``, in which case it is set to 0).
            eval_fn: Optional custom evaluation function. Receives the
                generative model and returns a scalar loss. If ``None``,
                evaluation uses the training loss on ``val_data``.
            io_handler: The IO handler which handles checkpointing
                and (specialised) logging. This is an optional argument.
                The default is `None`, which means that a default IO handler
                will be set up which does not include checkpointing but some
                very basic metrics logging.

        Raises:
            ValueError: If ``num_batches_per_eval_step`` is ``None`` and
                ``val_data`` is neither ``None`` nor sized.
        """
        self.train_data: Iterator[Batch] = train_data
        self.val_data: Iterator[Batch] | None = val_data
        self.loss_pipeline = loss_pipeline
        self.gen_model = gen_model
        self.optimizer_tx = optimizer_tx
        self.config = config
        self.eval_fn = eval_fn

        # Fall back to a default handler when none is given; only then attach our
        # default line-logger, so a caller-supplied handler keeps its own logging.
        self.io_handler: TrainingIOHandler = (
            io_handler if io_handler is not None else TrainingIOHandler()
        )
        if io_handler is None:
            self.io_handler.attach_logger(log_metrics_to_line)

        if num_batches_per_eval_step is not None:
            self.num_batches_per_eval_step = num_batches_per_eval_step
        elif isinstance(val_data, Sized):
            self.num_batches_per_eval_step = len(val_data)
        elif val_data is None:
            self.num_batches_per_eval_step = 0
        else:
            raise ValueError(
                "num_batches_per_eval_step must be provided when val_data has no __len__."
            )

        self.best_eval_loss = float("inf")
        self.best_eval_step = 0
        self._best_params: nnx.State | None = None

        self._init_training_and_dataset_states()

    def _init_training_and_dataset_states(self) -> None:
        """Split gen_model, initialise optimizer/EMA, initialise dataset state, and build JIT'd steps."""
        graphdef, params = nnx.split(self.gen_model, nnx.Param)
        opt_state = self.optimizer_tx.init(params)  # type: ignore[arg-type]  # nnx State is a valid optax param pytree
        ema_params = jax.tree.map(jnp.zeros_like, params)

        self.training_state = TrainingState(
            params=params,
            opt_state=opt_state,
            ema_params=ema_params,
            num_steps=jnp.array(0, dtype=jnp.int32),
            ema_decay=jnp.array(self.config.ema_decay, dtype=jnp.float32),
            rng_key=jr.PRNGKey(self.config.random_seed),
            graphdef=graphdef,
        )

        get_dataset_state = getattr(self.train_data, "get_state", None)
        self._get_dataset_state = (
            get_dataset_state if callable(get_dataset_state) else None
        )
        self.dataset_state = (
            self._get_dataset_state() if self._get_dataset_state is not None else None
        )

        self.training_state, self.dataset_state = self.io_handler.restore_checkpoint(
            self.training_state,
            self.dataset_state,
            restore_optimizer_state=self.io_handler.config.restore_optimizer_state,
        )

        # Re-apply the (possibly restored) cursor to the live iterator so a
        # resumed run continues the data stream where the checkpoint left off.
        # On a fresh start this just re-sets the iterator to its current state.
        set_dataset_state = getattr(self.train_data, "set_state", None)
        if self.dataset_state is not None and callable(set_dataset_state):
            set_dataset_state(self.dataset_state)

        self._train_step = _make_train_step(
            graphdef=graphdef,
            loss_pipeline=self.loss_pipeline,
            optimizer_tx=self.optimizer_tx,
            ema_decay=self.config.ema_decay,
            num_gradient_accumulation_steps=self.config.num_gradient_accumulation_steps,
        )
        self._eval_step = _make_eval_step(
            graphdef=graphdef,
            loss_pipeline=self.loss_pipeline,
        )

    def _eval_params(self) -> nnx.State:
        """Return the parameters to use for evaluation (EMA or raw)."""
        state = self.training_state
        if self.config.use_ema_params_for_eval and int(state.num_steps) > 0:
            return get_debiased_ema(
                state.ema_params,
                self.config.ema_decay,
                state.num_steps,
            )
        return state.params

    def _run_evaluation(self, key: PRNGKeyArray) -> tuple[nnx.State, PyTree[float]]:
        """Run evaluation and return ``(eval_params, metrics)``.

        ``metrics`` always contains ``"loss"``. Best-params bookkeeping is the
        caller's responsibility (see ``_update_best``).

        Args:
            key: PRNG key for the default (``val_data``-based) evaluation path.

        Returns:
            ``(eval_params, eval_metrics)``: the parameters evaluation ran with
            (EMA-corrected if ``use_ema_params_for_eval``) and the resulting
            metrics dict, always containing ``"loss"``.
        """
        eval_params = self._eval_params()

        eval_metrics: PyTree[float]
        if self.eval_fn is not None:
            eval_gen_model = nnx.merge(self.training_state.graphdef, eval_params)
            eval_metrics = {"loss": float(self.eval_fn(eval_gen_model))}
        elif self.val_data is not None:
            eval_metrics = self._default_eval(eval_params, key)
        else:
            eval_metrics = {"loss": float("inf")}

        return eval_params, eval_metrics

    def _update_best(
        self,
        step: int,
        eval_params: nnx.State,
        eval_metrics: PyTree[float],
    ) -> None:
        """If this eval beats the current best, stores the best parameters and the new best evaluation metrics.

        Args:
            step: The training step at which this evaluation occurred.
            eval_params: Parameters evaluation was run with, snapshotted if a
                new best.
            eval_metrics: Evaluation metrics dict, must contain ``"loss"``.
        """
        eval_loss = eval_metrics["loss"]
        if eval_loss < self.best_eval_loss:
            self.best_eval_loss = eval_loss
            self.best_eval_step = step
            self._best_params = jax.tree.map(jnp.copy, eval_params)
            logger.debug("New best step %d with eval loss: %.6f", step, eval_loss)

    def _default_eval(self, params: nnx.State, key: PRNGKeyArray) -> PyTree[float]:
        """Evaluate on val_data with the training loss.

        Per-batch metrics are kept as device arrays and reduced/transferred
        to host once per metric via a single ``jnp.mean`` call at the end.

        Args:
            params: The parameters to evaluate with.
            key: PRNG key, split once per evaluation batch.

        Returns:
            The metrics averaged over ``num_batches_per_eval_step`` batches,
            always containing ``"loss"``.

        Raises:
            ValueError: If no ``val_data`` was provided to the training loop.
        """
        val_data = self.val_data  # local binding so the None-check narrows (pyright doesn't narrow mutable attrs)
        if val_data is None:
            raise ValueError(
                "_default_eval requires val_data, but none was provided to the training loop."
            )
        per_batch_metrics: list[PyTree[Scalar]] = []

        for _ in range(self.num_batches_per_eval_step):
            batch = next(val_data)
            key, subkey = jr.split(key)
            per_batch_metrics.append(self._eval_step(params, batch, subkey))

        if not per_batch_metrics:
            return {"loss": float("inf")}

        return {
            k: float(jnp.mean(jnp.stack([m[k] for m in per_batch_metrics])))
            for k in per_batch_metrics[0].keys()
        }

    def _run_one_step(self, key: PRNGKeyArray) -> PyTree[Scalar]:
        """Run a single optimizer step, pulling batches from ``train_data``.

        With gradient accumulation, this consumes
        ``num_gradient_accumulation_steps`` batches and stacks them into a
        single pytree before invoking the JIT'd step.

        Args:
            key: PRNG key for the step.

        Returns:
            The step's metrics; ``self.training_state`` is updated in place.
        """
        n_accum = self.config.num_gradient_accumulation_steps
        if n_accum == 1:
            step_batch = next(self.train_data)
        else:
            buffer = [next(self.train_data) for _ in range(n_accum)]
            step_batch = jax.tree.map(lambda *xs: jnp.stack(xs), *buffer)

        new_state, metrics = self._train_step(self.training_state, step_batch, key)
        self.training_state = new_state
        return metrics

    def _log_window(
        self,
        step: int,
        window_metrics: list[PyTree[Scalar]],
        t_window_start: float,
        eval_key: PRNGKeyArray,
    ) -> None:
        """Block on the train window, aggregate, eval, update best, and log.

        Args:
            step: Current global training step (end of the window).
            window_metrics: Per-step metrics dicts accumulated since the last
                window, each containing ``"loss"``, ``"gradient_norm"``,
                ``"param_update_norm"``.
            t_window_start: ``time.perf_counter()`` timestamp marking the start
                of the training window, used to compute elapsed train time.
            eval_key: PRNG key for this window's evaluation pass.
        """
        jax.tree.map(lambda x: x.block_until_ready(), self.training_state)
        t_train = time.perf_counter() - t_window_start

        agg_train = {
            k: float(jnp.mean(jnp.stack([m[k] for m in window_metrics])))
            for k in window_metrics[0].keys()
        }

        t_eval_start = time.perf_counter()
        eval_params, eval_metrics = self._run_evaluation(eval_key)
        self._update_best(step, eval_params, eval_metrics)
        # Snapshot the live data cursor so the checkpoint records the position
        # reached at this step rather than the stale init-time cursor.
        if self._get_dataset_state is not None:
            self.dataset_state = self._get_dataset_state()
        self.io_handler.save_checkpoint(
            self.training_state,
            step,
            eval_metrics["loss"],
            dataset_state=self.dataset_state,
        )
        t_eval = time.perf_counter() - t_eval_start

        self.io_handler.log(
            LoggingCategory.TRAIN_METRICS,
            {
                "loss": agg_train["loss"],
                "gradient_norm": agg_train["gradient_norm"],
                "param_update_norm": agg_train["param_update_norm"],
            },
            step,
        )
        self.io_handler.log(
            LoggingCategory.EVAL_METRICS,
            {"loss": eval_metrics["loss"]},
            step,
        )
        self.io_handler.log(
            LoggingCategory.SYSTEM_METRICS,
            {"train_time_s": t_train, "eval_time_s": t_eval},
            step,
        )
        self.io_handler.log(
            LoggingCategory.BEST_MODEL,
            {"best_loss": self.best_eval_loss, "best_step": self.best_eval_step},
            step,
        )

    def run(self) -> None:
        """Run the full training loop. Final state is on ``self.training_state``.

        The PRNG stream is carried on ``training_state.rng_key`` rather than
        reseeded, so a resumed run continues the same noise/time stream. The
        key is re-stamped into ``training_state`` at each evaluation boundary
        (right before ``_log_window``, which is where the IO handler's
        ``save_checkpoint`` fires) so a restored checkpoint resumes from the
        exact key the loop would next split.

        Notes:
            - Checkpoints are written via ``self.io_handler.save_checkpoint``
              inside ``_log_window``, i.e. at the evaluation cadence. Box
              death between evaluations therefore loses at most one evaluation
              window of steps.
            - End-to-end bit-identity across a resume also requires a
              replayable ``train_data`` iterator. When the iterator exposes
              ``get_state``/``set_state`` (e.g. a grain dataset iterator), its
              cursor is checkpointed at each evaluation boundary and restored
              into the live iterator on resume, so the data stream continues
              where it left off. Iterators without those methods restart their
              stream on resume.
            - ``best_eval_loss``/``best_eval_step``/``best_params`` are not
              restored on resume: they track the best since *this process*
              started. The durable best checkpoint is tracked separately by
              the checkpoint store (``best_step()``).
        """
        num_steps = self.config.num_steps
        eval_every = self.config.eval_every_n_steps

        logger.info("Starting training loop (%d steps)...", num_steps)

        key = self.training_state.rng_key

        step = int(self.training_state.num_steps)

        if step > 0:
            self.io_handler.log(
                LoggingCategory.CLEANUP_AFTER_CKPT_RESTORATION, {}, step
            )

        # To avoid incorrect key splitting for resumed runs. Restarted runs
        # would have already undergone an eval step and thus would have already
        # performed key splitting.
        if self.config.run_eval_at_start and step == 0:
            key, eval_key = jr.split(key)
            eval_params, eval_metrics = self._run_evaluation(eval_key)
            self._update_best(step, eval_params, eval_metrics)
            self.io_handler.log(
                LoggingCategory.EVAL_METRICS, {"loss": eval_metrics["loss"]}, step
            )

        window_metrics: list[PyTree[Scalar]] = []
        t_window_start = time.perf_counter()

        while step < num_steps:
            key, step_key = jr.split(key)
            window_metrics.append(self._run_one_step(step_key))
            step += 1

            should_eval = (
                eval_every is not None and step % eval_every == 0
            ) or step == num_steps
            if not should_eval:
                continue

            key, eval_key = jr.split(key)
            # Stamp the post-split key so the checkpoint written inside
            # _log_window resumes from exactly here.
            self.training_state = self.training_state.replace(rng_key=key)
            self._log_window(step, window_metrics, t_window_start, eval_key)
            window_metrics = []
            t_window_start = time.perf_counter()

        # Belt-and-braces: keep the invariant "rng_key is the key to resume
        # from" even if the loop exits without a trailing evaluation.
        self.training_state = self.training_state.replace(rng_key=key)

        self.io_handler.wait_until_finished()

        logger.info(
            "Training complete. Best eval loss: %.6f (step %d)",
            self.best_eval_loss,
            self.best_eval_step,
        )

    @property
    def ema_model(self) -> GenerativeModel:
        """Reconstruct a live generative model with debiased EMA parameters."""
        ema_params = get_debiased_ema(
            self.training_state.ema_params,
            self.config.ema_decay,
            self.training_state.num_steps,
        )
        return nnx.merge(self.training_state.graphdef, ema_params)

    @property
    def best_params(self) -> nnx.State | None:
        """The parameters from the best evaluation step (EMA-corrected if enabled)."""
        return self._best_params

    def restore_model(self, params: nnx.State | None = None) -> GenerativeModel:
        """Merge params with the graphdef and return a live model.

        Args:
            params: Parameters to restore. Defaults to ``best_params``.

        Returns:
            A reconstructed ``GenerativeModel`` with the given parameters.

        Raises:
            ValueError: If ``params`` is ``None`` and no best parameters have
                been recorded (no evaluation has run).
        """
        params = params if params is not None else self._best_params
        if params is None:
            raise ValueError("No parameters to restore (no evaluation ran?).")
        return nnx.merge(self.training_state.graphdef, params)
