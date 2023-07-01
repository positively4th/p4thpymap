import pandas as pd
from collections.abc import Callable
from math import floor
from math import ceil
from math import log10
from os import linesep as eol
import re
import typing

from ..map import map
from . import rexp


def colWidth2TotWidth(colCount, colWidth, inverse=False):

    def _totWidth(colWidth):
        lineWidth = (colCount + 1) * (colWidth + 1) - 1
        return lineWidth

    def _colWidth(lineWidth):
        # lineWidth = (colCount + 1) * (colWidth + 1) - 1
        # 1 + lineWidth = (colCount + 1) * (colWidth + 1)
        # (lineWidth + 1) / (colCount + 1) = colWidth + 1
        # (lineWidth + 1) / (colCount + 1) - 1 = colWidth
        colWidth = floor((lineWidth + 1) / (colCount + 1) - 1)
        return colWidth

    return _colWidth(colWidth) if inverse else _totWidth(colWidth)


@map()
def format(df: pd.DataFrame, totWidth: int = None, colWidth: int = None):

    def floatFormat(val):
        res = val
        if isinstance(res, float):
            if res.is_integer():
                maxDec = 0
            else:
                nonDec = 1 if abs(
                    val) < 10 else ceil(0 + log10(abs(res)))
                maxDec = colWidth-nonDec-1
                maxDec = maxDec if res >= 0 else maxDec - 1
                maxDec = max(maxDec, 0)
            res = round(res, maxDec)
        res = str(res).rjust(colWidth, ' ')
        return res

    def cellFormat(val, colNr):
        sep = ' ' if colNr > 0 else ''
        if isinstance(val, (float, int)):
            return sep + floatFormat(val)
        res = str(val)
        if len(res) > colWidth:
            res = res[0:colWidth-3] + '...'
        return sep + res.ljust(colWidth, ' ')

    cols = len(df.columns)

    if colWidth is not None:
        totWidth = cols * colWidth if totWidth is None else totWidth

    if totWidth is not None:
        colWidth = colWidth2TotWidth(
            cols, totWidth, inverse=True) if colWidth is None else colWidth

    res = [
        ''.join([
            cellFormat(val, c) for c, val in enumerate([''] + df.columns.to_list())
        ])
    ]

    for label, row in df.iterrows():
        line = [cellFormat(label, 0)] + [
            cellFormat(val, c+1) for c, val in enumerate(row)
        ]
        res.append(''.join(line))
    return eol.join(res)


@map()
def keepRows(df: pd.DataFrame, filter):
    keeps = []
    for index, row in df.iterrows():
        if filter(row):
            keeps.append(row.tolist())
    return pd.DataFrame(keeps, columns=df.columns)


@map()
def keepColumns(df: pd.DataFrame,
                columnRe: re.Pattern = re.compile('.*'),
                filter: Callable[[pd.Series, str], pd.DataFrame] = lambda col, name: True) -> pd.DataFrame:
    keeps = {}
    for name, col in df.items():
        if not columnRe.match(name):
            continue
        if not filter(col, name):
            continue
        keeps[name] = keeps[name] if name in keeps else []
        keeps[name] = col.tolist()
    return pd.DataFrame(keeps)


@map()
def group(df: pd.DataFrame, grouper: Callable[[pd.Series, pd.Index], str]) -> dict:
    res = {}
    for index, row in df.iterrows():
        group = grouper(row, index)
        res[group] = res[group] if group in res else []
        res[group].append(row)

    return {group: pd.concat(rows, axis=1, ignore_index=True).transpose() for group, rows in res.items()}


@map()
def columns(df: pd.DataFrame, columnRE: str = '.*') -> tuple:
    for c in df.columns:
        if re.match(columnRE, c):
            yield c


@map()
def aggregate(df: pd.DataFrame, aggregates: typing.Iterable[str], columnRE=rexp.compile('.*')) -> pd.DataFrame:

    sdf = keepColumns(df, columnRe=columnRE)
    return sdf.aggregate(func=aggregates)


@map()
def correlation(df: pd.DataFrame, columnRE: re.Pattern = rexp.compile('.*')):

    sdf = keepColumns(df, columnRe=columnRE)
    return sdf.corr()
