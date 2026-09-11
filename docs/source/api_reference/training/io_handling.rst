.. _api_training_io:

Checkpointing and logging
=========================

IO handler
----------

.. autoclass:: stix.training.training_io_handler.TrainingIOHandler

.. autoclass:: stix.training.training_io_handler.TrainingIOHandlerConfig

.. autoclass:: stix.training.training_io_handler.LoggingCategory

.. autoexception:: stix.training.training_io_handler.CheckpointRestorationError

Checkpointer
------------

.. autoclass:: stix.training.checkpointer.Checkpointer

.. autoclass:: stix.training.checkpointer.CheckpointerConfig

.. autofunction:: stix.training.checkpointer.save_run_metadata

.. autofunction:: stix.training.checkpointer.read_run_metadata

EMA
---

.. autofunction:: stix.training.ema.get_debiased_ema

Loggers
-------

.. autofunction:: stix.training.training_loggers.log_metrics_to_line

.. autofunction:: stix.training.training_loggers.log_metrics_to_table

.. autofunction:: stix.training.training_loggers.convert_mse_to_rmse_in_logs
