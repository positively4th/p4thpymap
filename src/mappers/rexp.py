import re

from ..map import map


@map()
def any(patterns: tuple | list | set):
    return '|'.join(['(?:' + re + ')' for re in patterns])


@map()
def exactlify(pattern: str):
    return '^{}$'.format(re.escape(pattern))


@map()
def compile(pattern: str | re.Pattern) -> re.Pattern:
    return pattern if isinstance(pattern, re.Pattern) else re.compile(pattern)


def negate(pattern: str) -> str:

    if pattern[0] != '^':
        prefix = '^'
    else:
        prefix = ''

    if pattern[-1] != '$':
        suffix = '.*$'
    else:
        suffix = ''

    res = f"{ prefix }(?!.*{ pattern }){ suffix }"

    return res
