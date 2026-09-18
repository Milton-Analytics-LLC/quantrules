# SPDX-FileCopyrightText: 2026 Milton Analytics, LLC
# SPDX-License-Identifier: Apache-2.0
"""Exceptions raised by quantrules.

Every exception derives from [`QuantRulesError`][quantrules.exceptions.QuantRulesError],
so callers can catch everything this library raises with one `except` clause. The
concrete errors also derive from `ValueError`, so code that already handles `ValueError`
around numerical work keeps working unchanged.
"""

from __future__ import annotations

__all__ = ["ConfigurationError", "DataValidationError", "QuantRulesError"]


class QuantRulesError(Exception):
    """Base class for every error raised by quantrules."""


class ConfigurationError(QuantRulesError, ValueError):
    """A parameter or configuration value is outside its permitted range."""


class DataValidationError(QuantRulesError, ValueError):
    """An input series does not satisfy the quantrules data contract.

    See the *Data contract* section of the documentation for the rules that
    every input series must satisfy.
    """
