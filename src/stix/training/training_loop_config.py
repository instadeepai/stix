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

"""Training loop configuration."""

import pydantic
from typing_extensions import Annotated

PositiveInt = Annotated[int, pydantic.Field(gt=0)]
EMADecay = Annotated[float, pydantic.Field(gt=0.0, le=1.0)]


class TrainingLoopConfig(pydantic.BaseModel):
    """Pydantic config holding all settings related to the :class:`~stix.training.training_loop.TrainingLoop` class.

    Attributes:
        num_steps: Total number of optimizer updates to run.
        eval_every_n_steps: Run evaluation every N optimizer steps. If
            ``None``, evaluation only runs at the end of training (and
            optionally at the start, see ``run_eval_at_start``).
        num_gradient_accumulation_steps: Number of sub-batches to accumulate
            gradients over before applying an optimizer update. Default is 1.
        random_seed: A random seed.
        ema_decay: The EMA decay rate.
        use_ema_params_for_eval: Whether to use the EMA parameters for
            evaluation, set to ``True`` by default.
        run_eval_at_start: Whether to run an evaluation on the validation set
            before the first training step. ``True`` by default.
    """

    num_steps: PositiveInt
    eval_every_n_steps: PositiveInt | None = None
    num_gradient_accumulation_steps: PositiveInt = 1
    random_seed: int = 42
    ema_decay: EMADecay = 0.99
    use_ema_params_for_eval: bool = True
    run_eval_at_start: bool = True
