from functools import wraps

from .misc import isScalarByOp
from .misc import MapException


class MapAbort(Exception):

    def __init__(self, res):
        super().__init__('Map operation aborted withresult {}'.format(str(repr(res))))
        self.res = res

    @property
    def result(self):
        return self.res


def map(isScalar=None, abortTester=None, **_):

    def defAbortTester(result, input): return result

    _abortTester = defAbortTester if abortTester is None else abortTester

    def isScalarWrapper(op):

        _isScalar = isScalarByOp(op) if isScalar is None else isScalar

        @wraps(op)
        def do(input, *args, **kwargs):

            if _isScalar(input):
                return _abortTester(op(input, *args, **kwargs), input)
            if isinstance(input, (tuple)):
                return tuple([do(scalar, *args, **kwargs) for scalar in list(input)])
            if isinstance(input, (dict)):
                return {key: do(scalar, *args, **kwargs) for key, scalar in input.items()}
            if isinstance(input, (set)):
                return set([do(scalar, *args, **kwargs) for scalar in input])
            try:
                return [do(scalar, *args, **kwargs) for scalar in input]
            except MapAbort as e:
                raise e
            except TypeError:
                pass
            raise MapException(
                'Neither scalar nor iterable {}'.format(repr(input)))

        return do

    return isScalarWrapper
