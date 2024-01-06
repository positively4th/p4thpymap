import inspect
from typeguard import check_type, TypeCheckError


class MapException(Exception):
    pass


def isScalarByOp(op, argNr=0):

    def isScalar(inp):
        params = tuple(inspect.signature(op).parameters.values())
        assert len(params) > argNr
        scalarType = params[argNr].annotation
        assert scalarType is not inspect.Parameter.empty
        if isinstance(scalarType, str):
            return type(inp).__name__ == scalarType
        try:
            check_type(inp, scalarType)
        except TypeCheckError:
            return False
        return True

    return isScalar


def isScalar(inp):
    return not isinstance(inp, (list, tuple, dict))


def deferredIsScalar(isScalarOpts: tuple, input: any, args: tuple | list) -> callable:
    self = args[isScalarOpts[0]]
    method = isScalarOpts[1]
    method = getattr(self, method)
    _args = [input] if len(isScalarOpts) < 3 else [
        args[i]
        for i in isScalarOpts[2]
    ]
    return method(*_args)


def createIsScalarTypeGuarded(typeGetter: callable):

    def isScalarTypeGuarded(inp):
        try:
            check_type(inp, typeGetter())
        except TypeCheckError:
            return False

        return True

    return isScalarTypeGuarded


def isMethodOfInstance(func: callable) -> bool:
    return inspect.isfunction(func) and hasattr(func, '__self__') and inspect.isclass(func.__self__)


def replaceByIndex(input, argNr, args):
    res = args[0:argNr] + tuple([input]) + args[argNr+1:]
    return res
