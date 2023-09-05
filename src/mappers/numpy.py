import numpy as np

from ..map import map


@map()
def isValidNumber(scalar: float | int | str | None):
    try:
        float(scalar)
        return not np.isinf(scalar) and not np.isnan(scalar)
    except Exception as e:
        print(e)
        pass
    return False
