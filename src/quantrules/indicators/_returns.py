# SPDX-FileCopyrightText: 2026 Milton Analytics, LLC
# SPDX-License-Identifier: Apache-2.0
"""Return transforms shared inside the indicator layer.

Private for now: the public returns API lands in ``evaluate.returns`` in a
later phase. It lives here so the volatility estimators share one definition of
a return rather than each rederiving it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantrules._typing import FloatSeries


def _log_returns(price: FloatSeries) -> FloatSeries:
    r"""Natural-log returns, :math:`\ln(p_t / p_{t-1})`, with a leading `NaN`.

    Assumes ``price`` has already been validated against the data contract.
    """
    log_price = pd.Series(np.log(price.to_numpy()), index=price.index)
    return log_price.diff()
