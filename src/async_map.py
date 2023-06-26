from functools import wraps

from .misc import isScalarByOp
from .misc import MapException


def map(isScalar=None, abortTester=None, **_):

    async def defAbortTester(result, input): return result

    _abortTester = defAbortTester if abortTester is None else abortTester

    def isScalarWrapper(op):

        _isScalar = isScalarByOp(op) if isScalar is None else isScalar

        @wraps(op)
        async def do(input, *args, **kwargs):

            if _isScalar(input):
                return await _abortTester(await op(input, *args, **kwargs), input)

            if isinstance(input, (tuple)):
                return tuple([await do(scalar, *args, **kwargs) for scalar in list(input)])
            if isinstance(input, (list)):
                return [await do(scalar, *args, **kwargs) for scalar in input]
            if isinstance(input, (dict)):
                return dict(zip(input.keys(), [await do(input[key], *args, **kwargs) for key in input]))
            if isinstance(input, (set)):
                return set([await do(scalar, *args, **kwargs) for scalar in input])

            raise MapException(
                'Neither scalar nor iterable {}'.format(repr(input)))

        return do

    return isScalarWrapper
