import unittest
import asyncio
from src.async_map import map


class TestAsyncMap(unittest.IsolatedAsyncioTestCase):

    async def test_map_scalar(self):

        @map()
        async def square(val: int):
            await asyncio.sleep(0.1)
            return val * val

        @map(inputArgNr=1)
        async def square1(_: any, val: int):
            await asyncio.sleep(0.1)
            return val * val

        class Squarer():
            def isScalar(self, input, *args, **kwargs):
                return isinstance(input, (str, int, float, bool))

            @map(isScalar=(0, 'isScalar'), inputArgNr=2)
            async def square1(self, _: any, val: int):
                await asyncio.sleep(0.1)
                return val * val

        specs = [
            {'args': (2,), 'kwargs': {}, 'exp': 4},
            {'args': ([2, 3],), 'kwargs': {}, 'exp': [4, 9]},
            {'args': ({'two': 2, 'three': 3},), 'kwargs': {},
             'exp': {'two': 4, 'three': 9}},
            {'args': ({'two': [2, -2], 'three': {'three2': 3}},), 'kwargs': {},
             'exp': {'two': [4, 4], 'three': {'three2': 9}}},
        ]

        squarer = Squarer()

        for spec in specs:
            act = await square(*spec['args'], **spec['kwargs'])
            self.assertEqual(spec['exp'], act)
            act = await square1('a', *spec['args'], **spec['kwargs'])
            self.assertEqual(spec['exp'], act)
            act = await squarer.square1('a', *spec['args'], **spec['kwargs'])
            self.assertEqual(spec['exp'], act)

    async def test_map_list(self):

        def isScalar(x):
            return isinstance(x, list)

        @map(isScalar)
        async def squares_explicit(vals: list):
            await asyncio.sleep(0.1)
            return [val * val for val in vals]

        @map()
        async def squares_implicit(vals: list):
            await asyncio.sleep(0.1)
            return [val * val for val in vals]

        specs = [
            {'args': ([2, 3],),  'kwargs': {}, 'exp': [4, 9]},
            {'args': (([1, 2], [2, 3]),),  'kwargs': {},
             'exp': ([1, 4], [4, 9])},
            {'args': ({'two': [1, 2], 'three': [2, 3]},), 'kwargs': {},
             'exp': {'two': [1, 4], 'three': [4, 9]}},
        ]

        for spec in specs:
            act = await squares_explicit(*spec['args'], **spec['kwargs'])
            self.assertEqual(spec['exp'], act)
            act = await squares_implicit(*spec['args'], **spec['kwargs'])
            self.assertEqual(spec['exp'], act)
