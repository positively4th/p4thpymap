import unittest
from os import linesep

import pandas as pd
import pandas.testing as pdt

from src.mappers import pdf
from src.mappers import rexp


class TestPDF(unittest.TestCase):

    @staticmethod
    def df1():
        return pd.DataFrame.from_records([
            {'a': 1.0, 'b': -1.0},
            {'a': 2.0, 'b': -2.0},
            {'a': 3.0, 'b': -3.0},
        ])

    @staticmethod
    def df2():
        return pd.DataFrame.from_records([
            {'a': 1.5, 'b': -2.5, 'c': -1.0},
            {'a': 2.5, 'b': -3.5, 'c': 0.0},
            {'a': 3.5, 'b': -4.5, 'c': 1.0},
        ])

    @staticmethod
    def df3():
        return pd.DataFrame.from_records([
            {'a': -1, 'b': -1, 'c': 1.0, 'd': 1, },
            {'a': 0, 'b': 0, 'c': 0.0, 'd': 1, },
            {'a': 1, 'b': 1, 'c': -1.0, 'd': 1, },
        ])

    @staticmethod
    def df4():
        return pd.DataFrame.from_records([
            {'abcdef': 'OnePointOne', 'b': 1.1, 'd': 1.2345678, },
            {'abcdef': 'TwoPointTwo', 'b': 2.2, 'd': 2.3456789, },
            {'abcdef': 'ThreePointThree', 'b': 3.3, 'd': 3.4567890, },
        ], index=['row1', 'rowtwo', 'rowrowrow'])

    @staticmethod
    def df5():
        return pd.DataFrame.from_records([
            {'StatsBombXG': 39, 'Distance': 39},
            {'StatsBombXG': 0.016972719, 'Distance': 5.80517011},
            {'StatsBombXG': 0.105169345, 'Distance': 17.7173129},
            {'StatsBombXG': 0.057720438, 'Distance': 18.1592951},
            {'StatsBombXG': 0.46041265, 'Distance': 30.8058436},
        ], index=['count', 'min', 'mean', 'median', 'max'])

    def test_format4(self):
        df4 = TestPDF.df4()

        exp = linesep.join([
            '     a... b    d   ',
            'row1 O...  1.1 1.23',
            'r... T...  2.2 2.35',
            'r... T...  3.3 3.46',
        ])

        act = pdf.format(df4, colWidth=4)
        self.assertEquals(exp, act)
        act = pdf.format(df4, totWidth=(3+1)*(4+1)-1)
        self.assertEquals(exp, act)

        exp = linesep.join([
            '     row1 r... r...',
            'a... O... T... T...',
            'b     1.1  2.2  3.3',
            'd    1.23 2.35 3.46',
        ])

        act = pdf.format(df4.transpose(), colWidth=4)
        self.assertEquals(exp, act)

        act = pdf.format(df4.transpose(), totWidth=(3+1)*(4+1)-1)
        self.assertEquals(exp, act)

    def test_format5(self):
        df = TestPDF.df5()

        exp = linesep.join([
            '      St... Di...',
            'count  39.0  39.0',
            'min   0.017 5.805',
            'mean  0.105 17.72',
            'me... 0.058 18.16',
            'max    0.46 30.81',
        ])

        act = pdf.format(df, colWidth=5)
        self.assertEquals(exp, act)
        act = pdf.format(df, totWidth=(2+1)*(5+1)-1)
        self.assertEquals(exp, act)

        self.maxDiff = None
        exp = linesep.join([
            '                count           min             mean            median          max            ',
            'StatsBombXG                39.0     0.016972719     0.105169345     0.057720438      0.46041265',
            'Distance                   39.0      5.80517011      17.7173129      18.1592951      30.8058436',
        ])

        act = pdf.format(df.transpose(), colWidth=15)
        self.assertEquals(exp, act)
        act = pdf.format(df.transpose(), totWidth=(5+1)*(15+1)-1)
        self.assertEquals(exp, act)

    def test_correlation(self):
        df3 = TestPDF.df3()

        exp = pd.DataFrame.from_records([
            {'a': 1.0, 'b': 1.0, 'c': -1.0, 'd': float('nan'), },
            {'a': 1.0, 'b': 1.0, 'c': -1.0, 'd': float('nan'), },
            {'a': -1.0, 'b': -1.0, 'c': 1.0, 'd': float('nan'), },
            {'a': float('nan'), 'b': float('nan'),
             'c': float('nan'), 'd': float('nan'), },
        ], index=['x', 'b', 'c', 'd']).rename(index={'x': 'a'})
        act = pdf.correlation(df3, columnRE=rexp.compile(
            rexp.any(('a', 'b', 'c', 'd'))))
        pdt.assert_frame_equal(exp, act)

        exp = pd.DataFrame.from_records([
            {'a': 1.0, 'c': -1.0, },
            {'a': -1.0, 'c': 1.0, },
        ], index=['x', 'c']).rename(index={'x': 'a'})
        act = pdf.correlation(df3, columnRE=rexp.compile(
            rexp.any(('a', 'c'))))
        pdt.assert_frame_equal(exp, act)

    def test_aggregate(self):
        df2 = TestPDF.df2()

        aggregates = ['max', 'min']
        exp = pd.DataFrame.from_records([
            {'a': 3.5, 'c': 1.0},
            {'a': 1.5, 'c': -1.0},
        ], index=aggregates)
        act = pdf.aggregate(df2, aggregates=aggregates,
                            columnRE=rexp.compile(rexp.any(('a', 'c'))))
        pdt.assert_frame_equal(exp, act)

    def test_columns(self):
        self.assertEquals(('a',), tuple(pdf.columns(TestPDF.df2(), '^a$')))
        self.assertEquals(('b',), tuple(pdf.columns(TestPDF.df2(), '^b$')))
        self.assertEquals(('a', 'c'), tuple(
            pdf.columns(TestPDF.df2(), rexp.any(['^a$', '^c$']))))

    def test_keepRows(self):

        df1 = TestPDF.df1()
        df2 = TestPDF.df2()

        exp = pd.DataFrame.from_records([{'a': 2.0, 'b': -2.0}])
        actual = pdf.keepRows(df1, lambda r: r['a'] <= 2 and r['b'] <= -2)
        pdt.assert_frame_equal(exp, actual)

        exp = [
            pd.DataFrame.from_records([{'a': 2.0, 'b': -2.0}]),
            pd.DataFrame.from_records([{'a': 1.5, 'b': -2.5, 'c': -1.0}]),
        ]
        actual = pdf.keepRows(
            [df1, df2], lambda r: r['a'] <= 2 and r['b'] <= -2)
        self.assertAlmostEquals(2, len(actual))
        for i in range(len(exp)):
            pdt.assert_frame_equal(exp[i], actual[i])

    def test_keepCols(self):

        df2 = TestPDF.df2()

        exp = pd.DataFrame.from_records([
            {'b': -2.5},
            {'b': -3.5},
            {'b': -4.5},
        ])

        actual = pdf.keepColumns(df2,
                                 columnRe=rexp.compile(
                                     rexp.any(('^a$', '^b$'))),  # a or b
                                 filter=lambda col, name: col.max() < 2.5  # b or c
                                 )
        pdt.assert_frame_equal(exp, actual)

    def test_group(self):

        def oddEvenGrouper(row, i, *args, **kwargs):
            return 'even' if row['a'] % 2 == 0 else 'odd'

        df = self.df1()

        act = pdf.group(df, oddEvenGrouper)

        self.assertEquals(2, len(act))

        self.assertIn('even', act)
        expEven = pd.DataFrame.from_records([
            {'a': 2.0, 'b': -2.0},
        ])
        pdt.assert_frame_equal(expEven, act['even'])

        self.assertIn('odd', act)
        expOdd = pd.DataFrame.from_records([
            {'a': 1.0, 'b': -1.0},
            {'a': 3.0, 'b': -3.0},
        ])
        pdt.assert_frame_equal(expOdd, act['odd'])
