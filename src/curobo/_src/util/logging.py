"""Logging helpers matching pinned cuRoboV2."""

from __future__ import annotations

import functools
import logging
import sys
from typing import NoReturn, Type


def setup_logger(level="warning", logger_name: str = "curobo"):
    levels = {
        "info": logging.INFO,
        "debug": logging.DEBUG,
        "error": logging.ERROR,
        "warn": logging.WARNING,
        "warning": logging.WARNING,
    }
    if level not in levels:
        raise ValueError("Log level should be one of [info,debug, warn, error]")
    numeric_level = levels[level]
    logging.basicConfig(
        format="[%(levelname)s] [%(name)s] %(message)s", level=numeric_level
    )
    logging.getLogger(logger_name).setLevel(numeric_level)


def setup_curobo_logger(level="warning"):
    return setup_logger(level, "curobo")


def log_warn(txt: str, logger_name: str = "curobo", *args, **kwargs):
    logging.getLogger(logger_name).warning(txt, *args, **kwargs)


def log_debug(txt: str, logger_name: str = "curobo", *args, **kwargs):
    logging.getLogger(logger_name).debug(txt, *args, **kwargs)


def log_info(txt: str, logger_name: str = "curobo", *args, **kwargs):
    logging.getLogger(logger_name).info(txt, *args, **kwargs)


def log_and_raise(
    txt: str,
    logger_name: str = "curobo",
    exc_info=True,
    stack_info=False,
    stacklevel: int = 2,
    *args,
    exception_type: Type[Exception] = ValueError,
    **kwargs,
) -> NoReturn:
    logger = logging.getLogger(logger_name)
    log_kwargs = dict(exc_info=exc_info, stack_info=stack_info, **kwargs)
    if sys.version_info > (3, 7):
        log_kwargs["stacklevel"] = stacklevel
    logger.error(txt, *args, **log_kwargs)
    raise exception_type(txt)


def deprecated(reason):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            log_warn(f"DEPRECATED: {func.__name__} is deprecated. {reason}")
            return func(*args, **kwargs)

        wrapper.__deprecated__ = True
        return wrapper

    return decorator
