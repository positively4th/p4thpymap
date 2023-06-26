
from ..async_map import map
from ..map import MapAbort


def createAbortTester(tester: callable):

    async def abortTester(res, inp):
        if await tester(res, inp):
            raise MapAbort(res)
        return res

    return abortTester


async def all(_input, isTester: callable, isScalar=None):

    async def logicalNot(res, *_, **__): return not res

    _isTester = map(isScalar=isScalar,
                    abortTester=createAbortTester(logicalNot))(isTester)

    try:
        await _isTester(_input)
    except MapAbort:
        return False

    return True


async def any(input, isTester: callable, isScalar=None):

    async def logicalIdentity(res, *_, **__): return res

    _isTester = map(isScalar=isScalar,
                    abortTester=createAbortTester(logicalIdentity))(isTester)

    try:
        await _isTester(input)
    except MapAbort:
        return True

    return False


async def none(input, isTester: callable, isScalar=None):
    return not await any(input, isTester, isScalar)
