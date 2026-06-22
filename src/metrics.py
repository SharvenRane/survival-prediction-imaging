"""Concordance index for right censored survival data."""

import numpy as np


def concordance_index(time, event, risk) -> float:
    """Harrell concordance index.

    A pair of subjects is comparable when the one with the shorter time had an
    observed event. The pair is concordant when the subject who failed earlier
    carries the higher predicted risk. Tied risks count as half concordant.

    Args:
        time: array of survival or censoring times, shape (N,).
        event: array of event indicators (1 observed, 0 censored), shape (N,).
        risk: array of predicted risk scores where higher means more risk,
            shape (N,).

    Returns:
        Concordance index in [0, 1]. Returns 0.5 when no comparable pair exists.
    """
    time = np.asarray(time, dtype=np.float64).ravel()
    event = np.asarray(event, dtype=np.float64).ravel()
    risk = np.asarray(risk, dtype=np.float64).ravel()

    n = time.shape[0]
    concordant = 0.0
    permissible = 0.0

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # i is the earlier failure of the pair.
            if event[i] == 1 and time[i] < time[j]:
                permissible += 1.0
                if risk[i] > risk[j]:
                    concordant += 1.0
                elif risk[i] == risk[j]:
                    concordant += 0.5

    if permissible == 0.0:
        return 0.5
    return concordant / permissible
