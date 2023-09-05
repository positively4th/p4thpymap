import unittest
import re
from json import dumps

import src.mappers.rexp as rexp


class TestRExp(unittest.TestCase):

    def test_compile(self):

        self.assertTrue(isinstance(rexp.compile('a'), re.Pattern))
        self.assertTrue(isinstance(rexp.compile(re.compile('a')), re.Pattern))

        specs = [
            {'args': ['.*'], 'expected': re.compile('.*')},
            {'args': [['.', 'a']],
                'expected': [re.compile('.'), re.compile('a')]},
        ]

        for spec in specs:
            cre = rexp.compile(*spec['args'])

            self.assertEquals(spec['expected'], cre)

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

    def test_negate(self):

        specs = [
            {
                'pattern': 'a', 'subject': 'abc',
                'pattern': 'a', 'subject': 'def',
                'pattern': 'b', 'subject': 'abc',
                'pattern': 'b', 'subject': 'def',
                'pattern': 'c', 'subject': 'abc',
                'pattern': 'c', 'subject': 'def',

                'pattern': '^abc$', 'subject': 'abc',
                'pattern': '^abc$', 'subject': 'def',

                'pattern': '^a.*$', 'subject': 'abc',
                'pattern': '^a.*$', 'subject': 'def',

                'pattern': '^.*c$', 'subject': 'abc',
                'pattern': '^.*c$', 'subject': 'def',

                'pattern': '^.*b.*$', 'subject': 'abc',
                'pattern': '^.*b.*$', 'subject': 'def',
            }
        ]

        for spec in specs:
            isMatch = bool(
                re.search(pattern=spec['pattern'], string=spec['subject']))
            anitPattern = rexp.negate(spec['pattern'])
            print(isMatch, re.search(
                pattern=anitPattern, string=spec['subject']))
            self.assertEqual(not isMatch, bool(
                re.search(pattern=anitPattern, string=spec['subject'])), dumps(spec))
