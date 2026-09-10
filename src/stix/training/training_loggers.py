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

import logging
from copy import deepcopy
from typing import Any

import numpy as np
from rich import print as table_print
from rich.table import Table

from stix.training.training_io_handler import LoggingCategory

logger = logging.getLogger("stix")


def convert_mse_to_rmse_in_logs(to_log: dict[str, Any]) -> dict[str, Any]:
    """Convert metrics whose key contains ``mse_`` to ``rmse_`` by taking the square root.

    Applied at log time rather than during accumulation, because the square root must
    be taken after averaging rather than before.

    Args:
        to_log: The metrics dictionary.

    Returns:
        The metrics dictionary with any MSE entries converted to RMSE.
    """

    def _convert_value(key: str, value: Any) -> Any:
        return np.sqrt(value) if "mse_" in key else value

    return {
        key.replace("mse_", "rmse_"): _convert_value(key, value)
        for key, value in to_log.items()
    }


def _build_rich_table(
    metrics: dict[str, int | float], title: str, step_number: int, metric_color: str
) -> Table:
    """Build a rich ``Table`` listing metrics for display.

    Args:
        metrics: Mapping of metric name to value.
        title: Table title prefix; the step number is appended to it.
        step_number: Current step number, appended to the title.
        metric_color: Rich color/style name used for the metric-name column.

    Returns:
        A populated ``rich.table.Table`` ready to print.
    """
    table = Table(
        title=f"{title} at step {step_number}",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Metric")
    table.add_column("Value", justify="right")

    keys = [k.replace("_", " ").capitalize() for k in metrics.keys()]
    values = [v if isinstance(v, int) else f"{float(v):.3f}" for v in metrics.values()]

    for k, v in zip(keys, values):
        _k = f"[{metric_color}]{k}[/{metric_color}]"
        _v = f"[white]{v}[/white]"
        table.add_row(_k, _v)

    return table


def log_metrics_to_table(
    category: LoggingCategory, to_log: dict[str, Any], step_number: int
) -> None:
    """Logging function for the training loop which logs the metrics to a nice table.

    The table will be printed to the command line.

    This function also converts MSE metrics to RMSE before logging them.

    Args:
        category: The logging category describing what type of data is currently logged.
        to_log: The data to log (typically, the metrics).
        step_number: The current step number.
    """
    _to_log = convert_mse_to_rmse_in_logs(to_log)

    table = None
    if category == LoggingCategory.BEST_MODEL:
        table = _build_rich_table(_to_log, "Best model", step_number, "white")
    elif category == LoggingCategory.TRAIN_METRICS:
        table = _build_rich_table(_to_log, "Training set metrics", step_number, "cyan")
    elif category == LoggingCategory.EVAL_METRICS:
        table = _build_rich_table(
            _to_log, "Validation set metrics", step_number, "green"
        )
    elif category == LoggingCategory.TEST_METRICS:
        table = _build_rich_table(
            _to_log, "Test set metrics", step_number, "blue_violet"
        )
    elif category == LoggingCategory.SYSTEM_METRICS:
        table = _build_rich_table(_to_log, "System metrics", step_number, "yellow")

    if table is not None:
        table_print(table)


def _metrics_to_str(metrics: dict[str, Any]) -> str:
    result = ""
    for k, v in metrics.items():
        result += k.replace("_", " ").capitalize()
        result += " = "
        result += str(v if isinstance(v, int) else f"{float(v):.3f}")
        result += " | "

    if result == "":  # noqa: PLC1901
        return result
    return result[:-3]  # remove final separator


def log_metrics_to_line(
    category: LoggingCategory, to_log: dict[str, Any], step_number: int
) -> None:
    """Logging function for the training loop which logs the metrics to a single line.

    This function also converts MSE metrics to RMSE before logging them.

    Args:
        category: The logging category describing what type of data is currently logged.
        to_log: The data to log (typically, the metrics).
        step_number: The current step number.
    """
    _to_log = convert_mse_to_rmse_in_logs(to_log)

    if category == LoggingCategory.BEST_MODEL:
        logger.info(
            "Best model: Loss = %.3f | Best step = %s",
            _to_log["best_loss"],
            _to_log["best_step"],
        )

    elif category == LoggingCategory.TRAIN_METRICS:
        # Make sure loss is first
        to_log_copy = deepcopy(_to_log)
        to_log_modified = {"loss": to_log_copy.pop("loss")}
        to_log_modified.update(to_log_copy)

        logger.info("------------ Step %s ------------", step_number)
        logger.info("%-11s %s", "Training:", _metrics_to_str(to_log_modified))

    elif category == LoggingCategory.EVAL_METRICS:
        logger.info("Validation: %s", _metrics_to_str(_to_log))

    elif category == LoggingCategory.TEST_METRICS:
        logger.info("Testing: %s", _metrics_to_str(_to_log))

    elif category == LoggingCategory.SYSTEM_METRICS:
        logger.info("%-11s %s", "System:", _metrics_to_str(_to_log))
