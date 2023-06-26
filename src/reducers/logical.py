
from ..map import map
from ..map import MapAbort


def createAbortTester(tester: callable):

    def abortTester(res, inp):
        if tester(res, inp):
            raise MapAbort(res)
        return res

    return abortTester


def all(_input, isTester: callable, isScalar=None):

    _isTester = map(isScalar=isScalar,
                    abortTester=createAbortTester(lambda res, *_, **__: not res))(isTester)

    try:
        _isTester(_input)
    except MapAbort:
        return False

    return True


def any(input, isTester: callable, isScalar=None):
    _isTester = map(isScalar=isScalar,
                    abortTester=createAbortTester(lambda res, *_, **__: res))(isTester)

    try:
        _isTester(input)
    except MapAbort:
        return True

    return False


def none(input, isTester: callable, isScalar=None):
    return not any(input, isTester, isScalar)
