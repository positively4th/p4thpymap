from functools import wraps

from .misc import deferredIsScalar
from .misc import replaceByIndex
from .misc import isScalarByOp
from .misc import MapException


class MapAbort(Exception):

    def __init__(self, res):
        super().__init__('Map operation aborted withresult {}'.format(str(repr(res))))
        self.res = res

    @property
    def result(self):
        return self.res


def map(isScalar=None, inputArgNr=0, abortTester=None, instanceArgNr=None, **_):

    def defAbortTester(result, input): return result

    _abortTester = defAbortTester if abortTester is None else abortTester

    def isScalarWrapper(op):

        _isScalar = isScalarByOp(op, inputArgNr) \
            if isScalar is None else isScalar

        @wraps(op)
        def do(*args, **kwargs):
            input = args[inputArgNr]
            if _isScalar(input) if callable(_isScalar) else deferredIsScalar(_isScalar, input, args):
                return _abortTester(op(*args, **kwargs), input)
            if isinstance(input, (tuple)):
                return tuple([do(*replaceByIndex(scalar, inputArgNr, args), **kwargs) for scalar in list(input)])
            if isinstance(input, (dict)):
                return {key: do(*replaceByIndex(scalar, inputArgNr, args), **kwargs) for key, scalar in input.items()}
            if isinstance(input, (set)):
                return set([do(*replaceByIndex(scalar, inputArgNr, args), **kwargs) for scalar in input])
            try:
                return [do(*replaceByIndex(scalar, inputArgNr, args), **kwargs) for scalar in input]
            except MapAbort as e:
                raise e
            except TypeError:
                pass
            raise MapException(
                'Neither scalar nor iterable {}'.format(repr(input)))

        return do

    return isScalarWrapper
