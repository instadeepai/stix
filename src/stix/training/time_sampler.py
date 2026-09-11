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

from typing import Protocol

import jax.random as jr

from stix.typing import PRNGKeyArray, Time


class TimeSampler(Protocol):
    """Protocol for time sampling strategies."""

    def __call__(self, key: PRNGKeyArray) -> Time:
        """Sample a scalar time value.

        Args:
            key: PRNG key for the draw.

        Returns:
            A scalar time in the sampler's support.
        """
        ...


class UniformTimeSampler(TimeSampler):
    """Sample time uniformly from ``[t_min, t_max]``."""

    def __init__(self, t_min: float = 0.0, t_max: float = 1.0):
        """Initialise the uniform time sampler with ``t_min`` and ``t_max``.

        Args:
            t_min: The lower bound of the Uniform distribution.
            t_max: The upper bound of the Uniform distribution.
        """
        self.t_min = t_min
        self.t_max = t_max

    def __call__(self, key: PRNGKeyArray) -> Time:
        """Sample time uniformly from ``[t_min, t_max]``.

        Args:
            key: PRNG key for the draw.

        Returns:
            A scalar time in ``[t_min, t_max]``.
        """
        return jr.uniform(key, shape=(), minval=self.t_min, maxval=self.t_max)
