import inspect


class MapException(Exception):
    pass


def isScalarByOp(op):

    def isScalar(inp):
        params = tuple(inspect.signature(op).parameters.values())
        assert len(params) >= 1
        scalarType = params[0].annotation
        assert scalarType is not inspect.Parameter.empty
        return isinstance(inp, scalarType)

    return isScalar


def isScalar(inp):
    return not isinstance(inp, (list, tuple, dict))
