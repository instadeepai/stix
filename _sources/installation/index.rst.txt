.. _installation:

Installation
============

Install as a dependency (from PyPI)
-----------------------------------

CPU JAX is included by default. For GPU or TPU support, use the corresponding
extra:

.. code-block:: bash

    # CPU (default)
    pip install stix-ml

    # GPU (CUDA 12)
    pip install "stix-ml[gpu]"

    # TPU
    pip install "stix-ml[tpu]"

Or with ``uv``:

.. code-block:: bash

    uv add stix-ml             # CPU
    uv add "stix-ml[gpu]"      # GPU
    uv add "stix-ml[tpu]"      # TPU

Install for development
-----------------------

.. code-block:: bash

    git clone https://github.com/instadeepai/stix.git && cd stix
    uv sync --group dev
