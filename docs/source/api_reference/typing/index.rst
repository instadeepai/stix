.. _stix_typing:

Typing
======

The ``typing`` module defines the shared type vocabulary used throughout *stix*.

Batch
-----

.. autoclass:: stix.typing.data.Batch

Variable types
--------------

.. autoclass:: stix.typing.variable_type.VarType

Variable type aliases
---------------------

.. autodata:: stix.typing.variables.RawCtsVar

.. autodata:: stix.typing.variables.RawDisVar

.. autodata:: stix.typing.variables.RawVar

.. autodata:: stix.typing.variables.EmbeddedVar

.. autodata:: stix.typing.variables.NoiseVar

Variable pairs
--------------

.. autoclass:: stix.typing.variables.RawSourceTargetPair
    :no-members:

.. autoclass:: stix.typing.variables.EmbeddedSourceTargetPair
    :no-members:

.. autofunction:: stix.typing.variables.get_batch_size

Core type aliases
-----------------

.. automodule:: stix.typing.core
    :members:
