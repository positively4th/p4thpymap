from functools import wraps
from json import dumps

from .misc import deferredIsScalar
from .misc import replaceByIndex
from .misc import isScalarByOp
from .misc import MapException


def map(isScalar=None, inputArgNr=0, abortTester=None, jsonFallbackMap: dict = None, **_):

    async def defAbortTester(result, input): return result

    _abortTester = defAbortTester if abortTester is None else abortTester

    def isScalarWrapper(op):

        _isScalar = isScalarByOp(
            op, inputArgNr) if isScalar is None else isScalar

        @wraps(op)
        async def do(*args, **kwargs):

            input = args[inputArgNr]
            if _isScalar(input) if callable(_isScalar) else deferredIsScalar(_isScalar, input, args):
                return await _abortTester(await op(*args, **kwargs), input)

            if isinstance(input, (tuple)):
                return tuple([await do(*replaceByIndex(scalar, inputArgNr, args), **kwargs) for scalar in list(input)])
            if isinstance(input, (list)):
                return [await do(*replaceByIndex(scalar, inputArgNr, args), **kwargs) for scalar in input]
            if isinstance(input, (dict)):
                return dict(zip(input.keys(), [await do(*replaceByIndex(input[key], inputArgNr, args), **kwargs) for key in input]))
            if isinstance(input, (set)):
                return set([await do(*replaceByIndex(scalar, inputArgNr, args), **kwargs) for scalar in input])

            if jsonFallbackMap is not None:
                jsonInput = dumps(input)
                if jsonInput in jsonFallbackMap:
                    return jsonFallbackMap[jsonInput]

            raise MapException(
                'Neither scalar nor iterable {}'.format(repr(input)))

        return do

    return isScalarWrapper
