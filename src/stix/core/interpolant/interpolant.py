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

from abc import ABC, abstractmethod

from stix.core.generator import Generator
from stix.typing import (
    EmbeddedSourceTargetPair,
    EmbeddedVar,
    NoiseVar,
    PRNGKeyArray,
    Shape,
    Time,
    Var,
)


class Interpolant(ABC):
    r"""A base class for a generic interpolant.

    The general form of the interpolant is:

    :math:`z_t = I_t(z_{\mathrm{src}}, z_{\mathrm{tgt}},\epsilon)`

    with :math:`z_{\mathrm{src}}` and :math:`z_{\mathrm{tgt}}` the embedded source and target variables,
    and :math:`\epsilon` the noise variable. :math:`\epsilon` is *always* assumed to be independent
    of :math:`z_{\mathrm{src}}` and :math:`z_{\mathrm{tgt}}`.

    See :doc:`/introduction` for more details.
    """

    #: Sampling-time generator subclass returned for this interpolant family.
    generator_type: type[Generator]

    @abstractmethod
    def sample_noise(self, key: PRNGKeyArray, shape: Shape) -> NoiseVar:
        r"""Sample the interpolant's noise variable :math:`\epsilon`.

        Args:
            key: PRNG key used to draw the noise.
            shape: Shape of the noise variable to draw (typically the embedded
                target shape).

        Returns:
            A noise sample.
        """
        ...

    @abstractmethod
    def sample_initial_state(self, z_src: EmbeddedVar | None, epsilon: NoiseVar) -> Var:
        r"""Sampling-time initial state :math:`z_0`.

        At training time, :math:`z_0` is computed using
        :math:`z_0 = I_0(z_{\mathrm{src}}, z_{\mathrm{tgt}}, \epsilon)`.
        At sampling time, :math:`z_{\mathrm{tgt}}` is not available,
        and :math:`I_0` might still depend on :math:`z_{\mathrm{tgt}}`.

        Instead of feeding an arbitrary :math:`z_{\mathrm{tgt}}` to :math:`I_0`,
        this method is used as the sampling-time proxy to compute :math:`z_0`
        from :math:`z_{\mathrm{src}}` and :math:`\epsilon` only.

        Note: For one-sided interpolants, ``z_{\mathrm{src}}`` is always ``None``,
        so the initial state must be computed from :math:`\epsilon` only.

        Args:
            z_src: Embedded source :math:`z_{\mathrm{src}}`, or ``None`` for a
                one-sided interpolant.
            epsilon: The interpolant's noise variable :math:`\epsilon`.

        Returns:
            The initial embedded state :math:`z_0`.
        """
        ...

    @abstractmethod
    def interpolate(
        self, embedded_pairs: EmbeddedSourceTargetPair, t: Time, epsilon: NoiseVar
    ) -> EmbeddedVar:
        r"""Compute the interpolated state at time :math:`t`.

        .. math::
            z_t = I_t(z_{\mathrm{src}}, z_{\mathrm{tgt}}, \epsilon).

        Args:
            embedded_pairs: Embedded source-target pair :math:`(z_{\mathrm{src}}, z_{\mathrm{tgt}})`.
            t: Interpolation time.
            epsilon: The interpolant's noise variable :math:`\epsilon`.

        Returns:
            The interpolated state :math:`z_t`.
        """
        ...


class OneSidedInterpolant(Interpolant):
    r"""A base class for one-sided interpolant.

    The general form of the one-sided interpolant is:

    .. math::
        z_t = I_t(z_{\mathrm{tgt}},\epsilon)

    with :math:`z_{\mathrm{tgt}}` the embedded target variable,
    and :math:`\epsilon` the noise variable. As for the general interpolant,
    :math:`\epsilon` is always assumed to be independent of :math:`z_{\mathrm{tgt}}`.

    By construction, one-sided interpolants do not use any source variable :math:`z_{\mathrm{src}}`.

    At training time, :math:`z_t` is built from :math:`z_{\mathrm{tgt}}` and :math:`\epsilon`.

    At sampling time, :math:`z_{\mathrm{tgt}}` is not available, so
    :meth:`~stix.core.interpolant.Interpolant.sample_initial_state`
    maps :math:`\epsilon` to :math:`z_0` and ignores :math:`z_{\mathrm{src}}`.
    """
