import unittest
import re

import src.mappers.rexp as rexp


class TestRExp(unittest.TestCase):

    def test_compile(self):

        self.assertTrue(isinstance(rexp.compile('a'), re.Pattern))
        self.assertTrue(isinstance(rexp.compile(re.compile('a')), re.Pattern))

    def test_exactlify(self):

        specs = [
            {'pattern': 'a', 'expected': '^a$'},
            {'pattern': ['a', 'b'], 'expected': ['^a$', '^b$']},
            {'pattern': '?', 'expected': '^\?$'},
            {
                'pattern': {'?': '?', '+': '+'},
                'expected': {'?': '^\?$', '+': '^\+$'}
            },
        ]

        for spec in specs:
            self.assertEquals(spec['expected'], rexp.exactlify(
                spec['pattern']), msg=spec['pattern'])

    def test_any(self):

        specs = [
            {'pattern': ['a'], 'expected': '(?:a)'},
            {'pattern': ['a', 'b'], 'expected': '(?:a)|(?:b)'},
            {
                'pattern': {'A': ['a'], 'AB': ['a', 'b']},
                'expected': {'A': '(?:a)', 'AB': '(?:a)|(?:b)'}
            },
        ]

        for spec in specs:
            self.assertEquals(spec['expected'], rexp.any(
                spec['pattern']), msg=spec['pattern'])
