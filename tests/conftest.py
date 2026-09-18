# SPDX-FileCopyrightText: 2026 Milton Analytics, LLC
# SPDX-License-Identifier: Apache-2.0
"""Shared pytest configuration.

Hypothesis is pinned to a deterministic profile so that a green run on one
machine means a green run on every machine.
"""

from __future__ import annotations

from hypothesis import HealthCheck, settings

settings.register_profile(
    "quantrules",
    derandomize=True,
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile("quantrules")
