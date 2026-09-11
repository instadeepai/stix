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

import warnings
from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
from flax import nnx

from stix.typing import EmbeddedVar, RawDisVar, RawVar, Shape, Var


class Embedder(nnx.Module, ABC):
    r"""Base class for embedder and de-embedder maps between raw and embedded source/target pairs.

    Maps the raw data source/target pairs :math:`(x_{\mathrm{src}}, x_{\mathrm{tgt}})` to the embedded
    source/target pairs :math:`(z_{\mathrm{src}}, z_{\mathrm{tgt}})` and vice versa.
    """

    def __init__(
        self, dm_shape: Shape, embedding_shape: Shape, propagate_grads: bool = False
    ):
        """Initialise the embedder.

        Args:
            dm_shape: Shape of the modality's raw data.
            embedding_shape: Output shape of the embedding.
            propagate_grads: If ``False``, ``stop_gradient`` is applied after
                encoding. Defaults to ``False``.
        """
        self.dm_shape = dm_shape
        self.embedding_shape = embedding_shape
        self.propagate_grads = propagate_grads

    @abstractmethod
    def from_raw_to_embeddings(self, raw_var: RawVar) -> EmbeddedVar:
        """Encode raw data into the embedding space.

        Args:
            raw_var: The raw variable to encode.

        Returns:
            The encoded variable in embedding space.
        """
        pass

    @abstractmethod
    def from_embeddings_to_raw(self, embedded_var: EmbeddedVar) -> RawVar:
        """Decode embeddings back to raw data space.

        Args:
            embedded_var: The embedded variable to decode.

        Returns:
            The decoded variable in raw space.
        """
        pass


class IdentityEmbedder(Embedder):
    """Identity embedder: a no-op that returns the input as the embedded variable."""

    def __init__(self, dm_shape: Shape):
        """Initialise the identity embedder.

        Args:
            dm_shape: Shape of the modality's raw data, used as the embedding
                shape too.
        """
        super().__init__(
            dm_shape=dm_shape, embedding_shape=dm_shape, propagate_grads=False
        )

    def from_raw_to_embeddings(self, raw_var: RawVar) -> EmbeddedVar:
        """Return raw data unchanged.

        Args:
            raw_var: The raw variable.

        Returns:
            ``raw_var``, unchanged.
        """
        return raw_var

    def from_embeddings_to_raw(self, embedded_var: EmbeddedVar) -> RawVar:
        """Return embedded data unchanged.

        Args:
            embedded_var: The embedded variable.

        Returns:
            ``embedded_var``, unchanged.
        """
        return embedded_var


class LearnedDiscreteEmbedder(Embedder):
    """Learnable embedding table mapping discrete inputs to continuous embeddings.

    Decoding from embeddings to raw discrete indices proceeds as follows:

    1. The dot product of the embedding table and the embedded variable is
       computed and interpreted as logits.
    2. Decoding to an integer index is done using an argmax on the probabilities
       corresponding to the logits.

    .. warning::
        This embedder is **experimental and potentially unstable**. Learned
        discrete embeddings require careful stabilisation (e.g. noise schedules,
        codebook normalisation) that has not been implemented here — see CDCD
        (Dieleman et al., 2022) for discussion. Prefer ``OneHotDiscreteEmbedder``
        unless you have a specific reason to use learned embeddings.
    """

    def __init__(
        self,
        dm_shape: Shape,
        embedding_dim: int,
        propagate_grads: bool,
        rngs: nnx.Rngs,
    ):
        """Initialise the embedder with a learnable embedding table.

        Args:
            dm_shape: Shape of the modality's raw data; its last axis gives the
                number of categories.
            embedding_dim: Width of the learned embedding.
            propagate_grads: If ``False``, ``stop_gradient`` is applied after
                encoding.
            rngs: NNX RNG state used to initialise the embedding table.
        """
        warnings.warn(
            "LearnedDiscreteEmbedder uses learned embeddings without stabilisation "
            "(e.g. noise schedules, codebook normalisation — see CDCD, Dieleman "
            "et al. 2022). Prefer OneHotDiscreteEmbedder unless you have a "
            "specific reason to use learned embeddings.",
            stacklevel=2,
        )
        super().__init__(dm_shape, (embedding_dim,), propagate_grads)
        self.embedding_dim = embedding_dim
        self.num_states = dm_shape[-1]
        self.embed = nnx.Embed(self.num_states, embedding_dim, rngs=rngs)

    def from_raw_to_embeddings(self, raw_var: RawVar) -> EmbeddedVar:
        """Encode discrete indices (or one-hot vectors) into continuous embeddings.

        Args:
            raw_var: Integer indices, or one-hot vectors that are first argmaxed.

        Returns:
            The looked-up embeddings.
        """
        if not jnp.issubdtype(raw_var.dtype, jnp.integer):
            raw_var = jnp.argmax(raw_var, axis=-1)

        embedded_var = self.embed(raw_var)

        if not self.propagate_grads:
            embedded_var = jax.lax.stop_gradient(embedded_var)

        return embedded_var

    def from_embeddings_to_raw(self, embedded_var: EmbeddedVar) -> RawDisVar:
        """Decode embeddings to an integer index.

        Args:
            embedded_var: The embedded variable to decode.

        Returns:
            The argmax category index.
        """
        probs = self.from_embeddings_to_probs(embedded_var)
        return jnp.argmax(probs, axis=-1)

    def from_embeddings_to_logits(self, embedded_var: EmbeddedVar) -> Var:
        """Decode embeddings to logits over the categories.

        Args:
            embedded_var: The embedded variable to decode.

        Returns:
            Logits, the dot product of ``embedded_var`` with the embedding table.
        """
        embeddings = self.embed.embedding[...]
        return embedded_var @ embeddings.T

    def from_embeddings_to_probs(self, embedded_var: EmbeddedVar) -> Var:
        """Decode embeddings to a probability distribution over the categories.

        Args:
            embedded_var: The embedded variable to decode.

        Returns:
            Softmax probabilities over the categories.
        """
        logits = self.from_embeddings_to_logits(embedded_var)
        return jax.nn.softmax(logits, axis=-1)


class OneHotDiscreteEmbedder(Embedder):
    r"""Parameter-free one-hot embedder for discrete inputs.

    Maps integer indices to one-hot vectors, and passes existing one-hot
    inputs through unchanged. Decoding: ``argmax`` over the last axis.
    No learnable parameters. There are two construction modes,
    distinguished by whether ``num_states`` is passed explicitly:

    - **One-hot raw data** (``num_states=None``, the default): ``dm_shape``'s
      last axis *is* the category count, so the raw data are already one-hot
      vectors (or a single categorical index that one-hots to that width) and the
      embedded shape equals ``dm_shape``.
    - **Index-valued raw data** (``num_states`` given): the raw data are
      integer indices of shape ``dm_shape`` (e.g. a length-``L`` token sequence,
      ``dm_shape=(L,)``, or a scalar index, ``dm_shape=()``), and the one-hot axis
      is *appended*, so the embedded shape is ``(*dm_shape, num_states)``.
      This mode can be used, for instance, for masked diffusion, setting
      ``num_states = K + 1`` so the mask symbol (index ``K``) gets its own slot.
    """

    def __init__(
        self,
        dm_shape: Shape,
        num_states: int | None = None,
    ):
        """Initialise the one-hot embedder.

        Args:
            dm_shape: Data modality shape. For one-hot raw data its last dimension
                is the number of categories; for index-valued raw data it is the
                shape of the integer-index array (the one-hot axis is appended).
            num_states: One-hot width. Leave ``None`` for one-hot raw data
                (inferred as ``dm_shape[-1]``); pass it explicitly for
                index-valued raw data (``K`` for uniform, ``K + 1`` for masked
                diffusion).
        """
        if num_states is None:
            num_states = dm_shape[-1]
            embedding_shape: Shape = dm_shape
        else:
            embedding_shape = (*dm_shape, num_states)
        super().__init__(dm_shape, embedding_shape, False)
        self.num_states = num_states

    def from_raw_to_embeddings(self, raw_var: RawVar) -> EmbeddedVar:
        """Encode discrete indices to one-hot vectors.

        Args:
            raw_var: Integer indices, or already-one-hot vectors that are passed
                through unchanged.

        Returns:
            One-hot vectors of width ``num_states``.
        """
        if jnp.issubdtype(raw_var.dtype, jnp.integer):
            return jax.nn.one_hot(raw_var, self.num_states)
        # A non-integer input is taken to be one-hot already and passed through, so
        # its shape *is* the embedded shape. If that disagrees with the shape this
        # embedder declared, `num_states` was passed for raw data that are already
        # one-hot: the appended axis is never emitted.
        if raw_var.shape[-len(self.embedding_shape) :] != self.embedding_shape:
            raise ValueError(
                f"OneHotDiscreteEmbedder declared embedding_shape="
                f"{self.embedding_shape} but was given already-one-hot data of "
                f"shape {raw_var.shape}. Drop `num_states` for one-hot raw data "
                "(the width is then read from `dm_shape[-1]`); pass it only for "
                "integer-index raw data, where the one-hot axis is appended."
            )
        return raw_var

    def from_embeddings_to_raw(self, embedded_var: EmbeddedVar) -> RawDisVar:
        """Decode to an integer index using an argmax.

        Args:
            embedded_var: The one-hot embedded variable. We note that
                predictions may be passed in here which may not strictly be
                one-hot.

        Returns:
            The argmax category index over the last axis.
        """
        return jnp.argmax(embedded_var, axis=-1)
