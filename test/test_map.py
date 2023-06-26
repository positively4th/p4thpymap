import unittest

from src.map import map


class TestMap(unittest.TestCase):

    def test_map_scalar(self):

        @map()
        def square(val: int):
            return val * val

        specs = [
            {'args': (2,), 'kwargs': {}, 'exp': 4},
            {'args': ([2, 3],), 'kwargs': {}, 'exp': [4, 9]},
            {'args': ({'two': 2, 'three': 3},), 'kwargs': {},
             'exp': {'two': 4, 'three': 9}},
            {'args': ({'two': [2, -2], 'three': {'three2': 3}},), 'kwargs': {},
             'exp': {'two': [4, 4], 'three': {'three2': 9}}},
        ]

        for spec in specs:
            act = square(*spec['args'], **spec['kwargs'])
            self.assertEquals(spec['exp'], act)

    def test_map_list(self):

        def isScalar(x):
            return isinstance(x, list)

        @map(isScalar)
        def squares_explicit(vals: list):
            return [val * val for val in vals]

        @map()
        def squares_implicit(vals: list):
            return [val * val for val in vals]

        specs = [
            {'args': ([2, 3],),  'kwargs': {}, 'exp': [4, 9]},
            {'args': (([1, 2], [2, 3]),),  'kwargs': {},
             'exp': ([1, 4], [4, 9])},
            {'args': ({'two': [1, 2], 'three': [2, 3]},), 'kwargs': {},
             'exp': {'two': [1, 4], 'three': [4, 9]}},
        ]

        for spec in specs:
            act = squares_explicit(*spec['args'], **spec['kwargs'])
            self.assertEquals(spec['exp'], act)
            act = squares_implicit(*spec['args'], **spec['kwargs'])
            self.assertEquals(spec['exp'], act)
