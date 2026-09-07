"""Established logging helper import path."""

import logging

from curobo._src.util.logging import (
    log_debug, log_info, log_warn, setup_curobo_logger, setup_logger,
)


def log_error(message, logger_name="curobo", *args, **kwargs):
    logging.getLogger(logger_name).error(message, *args, **kwargs)


__all__ = [
    "log_debug", "log_error", "log_info", "log_warn", "setup_curobo_logger",
    "setup_logger",
]
