import unittest

from src.reducers import logical


def isNone(val: str | int | bool | None): return val is None


noneTuple = ('1', '2')
someTuple = ('1', None)
allTuple = (None, None)

noneList = ['1', '2']
someList = ['1', None]
allList = [None, None]

noneSet = set(['1', '2'])
someSet = set(['1', None])
allSet = set([None, None])

noneDict = {'A': '1', 'B': '2'}
someDict = {'A': '1', 'B': None}
allDict = {'A': None, 'B': None}


class TestLogical(unittest.TestCase):

    def test_all(self):

        specs = [
            {'subject': '1', 'isTester': isNone, 'expected': False},
            {'subject': None, 'isTester': isNone, 'expected': True},

            {'subject': noneTuple, 'isTester': isNone, 'expected': False},
            {'subject': someTuple, 'isTester': isNone, 'expected': False},
            {'subject': allTuple, 'isTester': isNone, 'expected': True},

            {'subject': noneList, 'isTester': isNone, 'expected': False},
            {'subject': someList, 'isTester': isNone, 'expected': False},
            {'subject': allList, 'isTester': isNone, 'expected': True},

            {'subject': noneSet, 'isTester': isNone, 'expected': False},
            {'subject': someSet, 'isTester': isNone, 'expected': False},
            {'subject': allSet, 'isTester': isNone, 'expected': True},

            {'subject': noneDict, 'isTester': isNone, 'expected': False},
            {'subject': someDict, 'isTester': isNone, 'expected': False},
            {'subject': allDict, 'isTester': isNone, 'expected': True},

            {'subject': [
                {
                    'A': noneList
                }
            ], 'isTester': isNone, 'expected': False},
            {'subject': [
                {
                    'A': someList
                }
            ], 'isTester': isNone, 'expected': False},
            {'subject': [
                {
                    'A': allList
                }
            ], 'isTester': isNone, 'expected': True},
        ]

        for spec in specs:
            self.assertEqual(spec['expected'], logical.all(
                spec['subject'], spec['isTester']), spec['subject'])

    def test_any(self):

        specs = [
            {'subject': '1', 'isTester': isNone, 'expected': False},
            {'subject': None, 'isTester': isNone, 'expected': True},

            {'subject': noneTuple, 'isTester': isNone, 'expected': False},
            {'subject': someTuple, 'isTester': isNone, 'expected': True},
            {'subject': allTuple, 'isTester': isNone, 'expected': True},

            {'subject': noneList, 'isTester': isNone, 'expected': False},
            {'subject': someList, 'isTester': isNone, 'expected': True},
            {'subject': allList, 'isTester': isNone, 'expected': True},

            {'subject': noneSet, 'isTester': isNone, 'expected': False},
            {'subject': someSet, 'isTester': isNone, 'expected': True},
            {'subject': allSet, 'isTester': isNone, 'expected': True},

            {'subject': noneDict, 'isTester': isNone, 'expected': False},
            {'subject': someDict, 'isTester': isNone, 'expected': True},
            {'subject': allDict, 'isTester': isNone, 'expected': True},

            {'subject': [
                {
                    'A': noneList
                }
            ], 'isTester': isNone, 'expected': False},
            {'subject': [
                {
                    'A': someList
                }
            ], 'isTester': isNone, 'expected': True},
            {'subject': [
                {
                    'A': allList
                }
            ], 'isTester': isNone, 'expected': True},
        ]

        for spec in specs:
            self.assertEqual(spec['expected'], logical.any(
                spec['subject'], spec['isTester']), spec['subject'])

    def test_none(self):

        specs = [
            {'subject': '1', 'isTester': isNone, 'expected': True},
            {'subject': None, 'isTester': isNone, 'expected': False},

            {'subject': noneTuple, 'isTester': isNone, 'expected': True},
            {'subject': someTuple, 'isTester': isNone, 'expected': False},
            {'subject': allTuple, 'isTester': isNone, 'expected': False},

            {'subject': noneList, 'isTester': isNone, 'expected': True},
            {'subject': someList, 'isTester': isNone, 'expected': False},
            {'subject': allList, 'isTester': isNone, 'expected': False},

            {'subject': noneSet, 'isTester': isNone, 'expected': True},
            {'subject': someSet, 'isTester': isNone, 'expected': False},
            {'subject': allSet, 'isTester': isNone, 'expected': False},

            {'subject': noneDict, 'isTester': isNone, 'expected': True},
            {'subject': someDict, 'isTester': isNone, 'expected': False},
            {'subject': allDict, 'isTester': isNone, 'expected': False},

            {'subject': [
                {
                    'A': noneList
                }
            ], 'isTester': isNone, 'expected': True},
            {'subject': [
                {
                    'A': someList
                }
            ], 'isTester': isNone, 'expected': False},
            {'subject': [
                {
                    'A': allList
                }
            ], 'isTester': isNone, 'expected': False},
        ]

        for spec in specs:
            self.assertEqual(spec['expected'], logical.none(
                spec['subject'], spec['isTester']), spec['subject'])
