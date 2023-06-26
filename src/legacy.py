from __future__ import print_function
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import sqlite3
import sys
import json
import uuid

from io import StringIO
import pandas as pd


if __name__ == "__main__":
    from p4thtensor import P4thTensor
else:
    from .p4thtensor import P4thTensor

import re

class P4thPDFDeprecatedError(Exception):
    pass

class P4thPDF:

    _idColumn = '_id'

    @staticmethod
    def isIterable(o):
        try:
            #iterator = Tools.keyValIter(o)
            #iterator = o.iter()
            len(o)
        except TypeError:
            return False
        return True
    
    @staticmethod
    def listify(l, noneValue=None):
        if l is None:
            return noneValue
        if isinstance(l, str):
            return [l]
        try:
            iterator = iter(l)
            return l
        except TypeError:
            return [l]

    @staticmethod
    def eval(f, *args, **kwargs):
        if callable(f):
            return f(*args, **kwargs)
        return f

    @staticmethod
    def uniform(min=0, max=1, count=None):
        res = np.random.uniform(0, 1, 1 if count == None else count)
        return res[0] if count == None else res
    
    @staticmethod
    def bool2IntTransform(value):
        return int(value) if isinstance(value, bool) else value
    
    @staticmethod
    def uuid():
        return str(uuid.uuid4())
    
    @staticmethod
    def idColumnRe(idColumn=None):
        idC = P4thPDF._idColumn if idColumn == None else idColumn 
        return '^' + idC + '$'

    @staticmethod
    def notIdColumnRe(idColumn=None):
        idC = P4thPDF._idColumn if idColumn == None else idColumn 
        return '^(?!(?:' + idC + ')$).*$'

    @staticmethod
    def orIdColumnRe(re, idColumn=None):
        idCRe = P4thPDF.idColumnRe(idColumn) 
        return '|'.join(['(?:' + re + ')', '(?:' + idCRe + ')'])

    @staticmethod
    def orRe(*kargs):
        return '|'.join(['(?:' + re + ')' for re in kargs])

    @staticmethod
    def exactlifyRe(res):
        if not isinstance(res, str) and P4thPDF.isIterable(res):
        #if isinstance(res, (list, tuple)):
            return [P4thPDF.exactlifyRe(re) for re in res]
        return '^{}$'.format(res)

    @staticmethod
    def buildQuery(q, p=[], where=None, limit=None):

        if where != None:
            q = q + " WHERE " + where

        #q = q + " ORDER BY rowid DESC";

        if limit != None:
            q = q + " LIMIT ?"
            p.append(limit);

        return q,p

    def __init__(self, df=None):
        self._state = df
        
    def loadRows(self, rows, T=lambda row: row, ctxT=None, idColumn=None):
        ctxT = T(True, ctxT)
        for row in rows:
            res = T(row, ctxT)
            if False == res: 
                break
            ctxT = res

        rows = T(False, ctxT)
        if not isinstance(rows, dict):
            return rows
        df = pd.DataFrame(**rows)
        
        self._state = df

        if idColumn != False:
            self.identify(self.idColumn if idColumn == None else idColumn)
        return self


        
    def loadSQL(self, db, q, p=[], T=lambda row: row, ctxT=None, where=None, limit=None, idColumn=None):
        q,p = P4thPDF.buildQuery(q, p, where=where, limit=limit)
        c = db.cursor()
        r = c.execute(q, p)
        
        return self.loadRows(r, T, ctxT, idColumn)
        

    def ensureColumns(self, colValueMap):
        def do(df, colValueMap):
            for key, val in colValueMap.items():
                if not key in df:
                    df[key] = val
            return df

        self._state = P4thPDF._apply(self.state, do, colValueMap)
        return self
        
    def transformColumns(self, colReTransformMap):
        def do(df):
            ddf = df.copy(deep=True)
            for col in ddf.columns:
                for pat, T in colReTransformMap.items():
                    if re.search(pat, col):
                        ddf[col] = ddf[col].apply(T)
            return ddf

        self._state = P4thPDF._apply(self.state, do)
        return self
        
    def keepColumns(self, reFilter='.*', idColumn=None, renameMap={}):

        def do(df0, reFilter, idColumn):
            df = df0.filter(regex=reFilter)
            if idColumn:
                pass

            #pd.concat([df,df0], axis=1, join_axes=[df.index])

            rm = {
                old: new for old, new in renameMap.items() if old in df.columns
            }
            #return df
            return df.rename(columns=rm)

        self._state = P4thPDF._apply(self.state, do, reFilter, self.idColumn if idColumn == None else idColumn)
        return self

    def addColumns(self, nameFMap, replace=True):

        def do(df0, nameFMap):
            d = {}
            for key, f in nameFMap.items():
                d[key] = [None] * len(df0)
            for i, row in df0.iterrows():
                for key,f in nameFMap.items():
                    d[key][i] = P4thPDF.eval(f, row)

            df = df0.copy(deep=False)
            if replace:
                df = df0.drop(list(d.keys()), errors='ignore')

            df = pd.concat([df, pd.DataFrame(d)], axis=1)
            return df

        self._state = P4thPDF._apply(self.state, do, nameFMap)
        return self

    def hasColumns(self, reFilter='.*'):

        def do(df0, reFilter):
            df = df0.filter(regex=reFilter)
            return len(df.columns) > 0

        self._state = P4thPDF._apply(self.state, do, reFilter)
        return self

    def keepRows(self, filterF):
        
        def do(df, filterF):
            keeps = []
            for index, row in df.iterrows():
                if filterF(row):
                    keeps.append(row.tolist())
            return pd.DataFrame(keeps, columns=df.columns)

        self._state = P4thPDF._apply(self.state, do, filterF)
        return self

        
    def identify(self, column=None):

        def do(df0, column):
            ids = [P4thPDF.uuid() for x in range(len(df0))]
            df0[column] = ids
            return df0

        self._state = P4thPDF._apply(self.state, do, P4thPDF._idColumn if column == None else column)
        return self
        
    def catsAsDummies(self, column, cats=None, refCat=None, reNamePattern=r'(.*)', reNameReplace=r'\1', join=False, typer=None):

        def do(df, crePattern):
            _cats = set(df[column] if cats is None else cats)
            _refCat = _cats.pop if refCat is None else refCat
            _cats = _cats.difference([refCat])
            colCatMap = {crePattern.sub(reNameReplace, cat): cat for cat in _cats}
            res = pd.DataFrame()
            for col, cat in colCatMap.items():
                res[col] = df[column] == cat
                if typer is not None:
                    res[col] = res[col].apply(typer)

            if join:
                res = pd.concat([df, res], axis=1, join_axes=[df.index])
            return res

        self._state = P4thPDF._apply(self.state, do, re.compile(reNamePattern) if isinstance(reNamePattern, str) else rePattern)
        return self
        
    def collapseDummies(self, reFilter='.*', catReplace=None, column=None, join=False, idColumn=False):

        def do(state, reFilter, column, join, idColumn):

            def collapse(row):
                nonlocal df
                nonlocal idColumn
                nonlocal catReplace
                
                collapsed = [ col if catReplace == None else reFilter.sub(catReplace, col)  for i,col in enumerate(df.columns) if (idColumn == False or col != idColumn) and row[i] > 0 ]
                return collapsed
        

            reFilter = re.compile(reFilter)
            df = state.filter(regex=reFilter)
            columns = [column if column != None else reFilter]
            df = pd.DataFrame(df.apply(collapse, axis=1), columns=columns)
            if idColumn:
                df = pd.concat([df, state[idColumn]], axis=1, join_axes=[df.index])
            if join:
                df = P4thPDF(state).join(df, idColumn).state
            return df
        
        self._state = P4thPDF._apply(self.state, do, \
                                     reFilter if idColumn == False else P4thPDF.orIdColumnRe(reFilter, idColumn), column, join, \
                                     P4thPDF._idColumn if idColumn == None else idColumn)
        return self
                        
    def dummiesAsOneHot(self, reFilter='.*', noneCat='none', noneReplace=r'\1', sepReplace=r'\2', catReplace=r'\3', typer=bool, join=False, idColumn=None):

        def do(df0, reFilter, noneCat, catReplace, noneReplace, join, idColumn):

            reFilter = re.compile(reFilter)
            df = P4thPDF(df0).keepColumns(reFilter, idColumn=False).state
            noneColumn = False
            manyRows = {}
            for index, row in df.iterrows():
                ones=[]
                lastCol = None
                for t in row.iteritems():
                    col, val = t
                    if val > 0:
                        lastCol = col
                        ones.append(col)

                if len(ones) < 1 and not noneColumn:
                    noneColumn = reFilter.sub(noneReplace, col) + reFilter.sub(sepReplace, col) + noneCat 

                if len(ones) > 1:
                    label = [reFilter.sub(noneReplace, lastCol)] + [reFilter.sub(catReplace, col) for col in ones]
                    label = reFilter.sub(sepReplace, lastCol).join(label)
                    manyRows[label] = ones

            columns = [noneColumn] if noneColumn else []
            columns = columns + \
                       [col for col in list(manyRows.keys())]
            dfNew = pd.DataFrame([], columns=columns)
            for index, row in df.iterrows():
                newRow = []
                if noneColumn:
                    newRow.append(typer(row.sum() < 1))
                for label, cols in manyRows.items():
                    bin = True
                    for col in cols:
                        bin = bin and row[col]
                    newRow.append(typer(bin))
                dfNew.loc[index] = newRow
                
            for index, row in df.iterrows():
                for label, cols in manyRows.items():
                    if dfNew.loc[index][label]:
                        for col in cols:
                            row[col] = typer(False)

            df = pd.concat([df, dfNew], axis=1, join_axes=[df.index])

            if idColumn:
                df = pd.concat([df, df0[idColumn]], axis=1, join_axes=[df.index])
            if join:
                df = P4thPDF(df).join(df0, idColumn).state
            return df

        idc = P4thPDF._idColumn if idColumn == None else idColumn
        self._state = P4thPDF._apply(self.state, do, \
                                     reFilter, noneCat, catReplace, noneReplace, join, \
                                     idc)
        return self
                        
    def plotCategories(self, catColumn):

        def do(state, catColumn):
            grouped = state.groupby(lambda i: '+'.join(state[catColumn][i]), squeeze=True)
            return grouped.count().plot(kind='bar')

        self._state = P4thPDF._apply(self.state, do, catColumn)
        return self
                        
    def partitionRows(self, categorizer, *args):

        def do(df, labelWeightMap):

            randCol = P4thPDF.uuid()

            tot = sum(labelWeightMap.values())
            state = {label: weight / tot for label, weight in labelWeightMap.items()}

            rdf = pd.DataFrame(np.random.uniform(0, 1, len(df)), columns=[randCol])
            df = pd.concat([df,rdf], axis=1, join_axes=[df.index])
            
            min = [0] * len(df)
            for label in state.keys():
                max = [min[0] + state[label]] * len(df)
                state[label] = df[(df[randCol] >= min) & (df[randCol] < max)]
                min = max 
                state[label] = state[label].drop(columns=[randCol])

            return state

        def doCategorizer(df, categorizer, *args):

            catCol = P4thPDF.uuid()

            cats = [''] * len(df);
            for i, row in df.iterrows():
                cats[i] = categorizer(row, i, *args)

            #tot = sum(labelWeightMap.values())
            #state = {label: weight / tot for label, weight in labelWeightMap.items()}

            rdf = pd.DataFrame(cats, columns=[catCol])
            df = pd.concat([df,rdf], axis=1, join_axes=[df.index])

            state = {}
            for cat in cats:
                state[cat] = df[(df[catCol] == cat)]
                state[cat] = state[cat].drop(columns=[catCol])
                state[cat].index = range(len(state[cat]))
            return state

        if isinstance(categorizer, dict):
            if dict(categorizer) == {}:
                return self
            self._state = P4thPDF._apply(self.state, do, categorizer)
        else:
            self._state = P4thPDF._apply(self.state, doCategorizer, categorizer)
        return self
                        

    def partitionColumns(self, labelReMap, idColumn=None):

        def reOr(res):
            return '|'.join(['(' + re + ')' for re in res])
            
        def do(df, labelReMap, idColumn):
           
            randCol = P4thPDF.uuid()
            df = {label: df.filter(regex=reFilter) for label, reFilter in labelReMap.items()}

            return df

        idC = P4thPDF.idColumnRe() if idColumn == None else idColumn
        if idC != False:
            map = {label: P4thPDF.orIdColumnRe(reFilter) for label, reFilter in labelReMap.items()}
        else:
            map = labelReMap

        self._state = P4thPDF._apply(self.state, do, map, idC)
        return self

    def values(self, idColumn=False):

        def do(df, idColumn):
            if not idColumn:
                tmp = df.filter(regex=P4thPDF.notIdColumnRe())
            else:
                tmp = df
            return tmp.values

        self._state = P4thPDF._apply(self.state, do, idColumn)
        return self
    
    def columns(self, reFilter=r'.*', idColumn=False):

        def do(df, reFilter):
            return [c for c in df.columns if reFilter.match(c)] 

        idC = P4thPDF.idColumn() if idColumn == None else idColumn

        reC = reFilter if idColumn == False else P4thPDF.orIdColumnRe(reFilter, idColumn) 
        self._state = P4thPDF._apply(self.state, do, re.compile(reFilter))
        return self

    def sortColumns(self):

        def do(df0):
            return df0.reindex(columns=sorted(df0.columns))

        self._state = P4thPDF._apply(self.state, do)
        return self

    def sortRows(self, by, ascending=True):

        def do(df0, reCF):
            
            res = df0.sort_values(by=by, ascending=True)
            res.reset_index(drop=True, inplace=True)
            return res
            
        self._state = P4thPDF._apply(self.state, do, by)
        return self


    def mostLikely(self, reColFilter=None, reRename=None, idColumn=None, join=False):

        def do(df, reColFilter, reRename, idColumn, join):
            tmp = df.filter(regex=reColFilter)
            tmp = tmp.apply(lambda ps: 1 * (ps >= ps.max()), axis=1)
            if idColumn != False:
                tmp = pd.concat([tmp, df[idColumn]], axis=1, join_axes=[tmp.index])
            if reRename != None:
                tmp = P4thPDF(tmp).rename(reColFilter, reRename).state
            if join:
                tmp = P4thPDF(df).join(tmp, idColumn).state
            return tmp

        self._state = P4thPDF._apply(self.state, do, reColFilter, reRename, self.idColumn if idColumn == None else idColumn, join)
        return self
    
    def likely(self, reColFilter=None, threshold=0.5, idColumn=None):

        def do(df, reColFilter, threshold, idColumn):
            tmp = df.filter(regex=reColFilter)
            tmp = tmp.apply(lambda p: 1 * (p > threshold))
            if idColumn != False:
                tmp = pd.concat([tmp, df[idColumn]], axis=1, join_axes=[tmp.index])
            return tmp

        self._state = P4thPDF._apply(self.state, do, reColFilter, threshold, self.idColumn if idColumn == None else idColumn)
        return self
    
    def join0(self, dfRight, idColumn=False):

        def do(df, dfRight, idColumn, rsuffix):
            colsToKeep = list(set(df.columns.tolist()).union(set(dfRight.columns.tolist())))
            if idColumn != False:
                df.set_index(idColumn)
                dfRight.set_index(idColumn)
            return df.join(dfRight, rsuffix=rsuffix)[colsToKeep]
        

        self._state = P4thPDF._apply(self.state, do,
                                    dfRight,
                                    self.idColumn if idColumn == None else idColumn,
                                    P4thPDF.uuid())
        return self

    def join(self, dfRight, joinType='inner', on=None, reRightColFilter='.*', rightSuffix='', rOn=None, *args, **kwargs):

        def do(df, on, rOn):

            onSuffix = P4thPDF.uuid()
            rCols = set(P4thPDF(dfRight).columns(reRightColFilter).state)

            rdf = { col+rightSuffix: dfRight[col] for col in rCols.difference(set(rOn)) }
            for col in rOn:
                rdf[col + onSuffix] = dfRight[col]
            rOn = [col+onSuffix for col in rOn]
            rdf = pd.DataFrame(rdf);

            mdf = df.merge(rdf, left_on=on, right_on=rOn, how=joinType, suffixes=[False, False])

            mdf.drop(rOn, axis=1, inplace=True);
            mdf.reset_index(inplace=True, drop=True)

            return mdf
        

        if not on:
            print(joinType, args)
            print('Warning: join without on is depricated')
            return self.join0(dfRight, joinType)


        on = P4thPDF.listify(on, None)
        self._state = P4thPDF._apply(self.state, do,
                                     [self.idColumn] if on is None else on,
                                     on if rOn is None else rOn)
        return self

    
    def lagged(self, reFilter=r'.*', groupBy=None, prefix='', suffix='', lag=0, *args, **kwargs):

        def do(df):
            sdf = df.copy(deep=True)

            cols = P4thPDF(df).columns(reFilter).state
            if groupBy is not None:
                gdf = sdf.groupby(by=groupBy)
            else:
                gdf = sdf.copy()
            for column in cols:
                #print(column)
                sdf[prefix+column+suffix] = gdf[column].shift(-lag).reset_index(0, drop=True)

            return sdf

        if 'sortBy' in kwargs:
            raise P4thPDFDeprecatedError('sortBy in lagged is deprecated!') 

        self._state = P4thPDF._apply(self.state, do)
        return self

    
    def rolling(self, reFilter=r'.*', groupBy=None, prefix='', suffix='', window=1, *args, **kwargs):

        def do(df):
            sdf = df.copy(deep=True)

            cols = P4thPDF(df).columns(reFilter).state
            if groupBy is not None:
                gdf = sdf.groupby(by=groupBy)
            else:
                gdf = sdf.copy()
            for column in cols:
                sdf[prefix+column+suffix] = gdf[column].rolling(window=window, *args, **kwargs).mean().reset_index(0, drop=True)

            #sdf.reset_index(drop=True, inplace=True)
            #print(sdf)
            return sdf

        if 'sortBy' in kwargs:
            raise P4thPDFDeprecatedError('sortBy in rolling is deprecated!') 
        self._state = P4thPDF._apply(self.state, do)
        return self

    def group(self, keys, aggregates={}, countLabel=None, reColumnFilter=r'.*', joinType=None):

        def do(df):
            reCols = P4thPDF.orRe(*([reColumnFilter] + P4thPDF.exactlifyRe(keys)))
            indexed = df.filter(regex=reCols).groupby(keys)

            if len(aggregates) > 0:
                res = indexed.aggregate(aggregates.values()).reset_index()
                aggNames = {v.__name__: k for k,v in aggregates.items()}
                res.columns = [vs[0] + aggNames[vs[1]] if vs[1] in aggNames else vs[0] for vs in res.columns.values]
            else:
                res = P4thPDF(df.drop_duplicates(subset=keys)).keepColumns(P4thPDF.orRe(*[r'^{}$'.format(c) for c in keys])).state
            if countLabel:
                cres = df.filter(items=keys)
                cres[countLabel] = 1.0
                cres = cres.groupby(keys).aggregate(np.sum).reset_index()
                res = res.set_index(keys).join(cres.set_index(keys), on=keys).reset_index()
            #print(res)

            if joinType:
                res = P4thPDF(df).join(res, joinType=joinType, on=keys).state

            return res

        self._state = P4thPDF._apply(self.state, do)
        return self
        
    def rename(self, needleRe, replaceRe):

        def do(df, needleRe, replaceRe):

            def mapper(label):
                return re.sub(needleRe, replaceRe, str(label))

            return df.rename(mapper=mapper, axis='columns')


        self._state = P4thPDF._apply(self.state, do, re.compile(needleRe), replaceRe)
        return self

    def transform(self, T):

        def do(df, T):
            return df.applymap(T)

        self._state = P4thPDF._apply(self.state, do, T)
        return self

    def diff(self, leftRe, rightRe, reRepl=r'diff_\1', idColumn=None):
        
        def do(df, leftRe, rightRe, reRepl, idColumn):
            
            lCols = P4thPDF(df).columns(leftRe, idColumn=False).state
            rCols = P4thPDF(df).columns(rightRe, idColumn=False).state
            assert len(lCols) == len(rCols)
            lCols = {leftRe.sub(reRepl, col): col for i,col in enumerate(lCols)}
            rCols = {rightRe.sub(reRepl, col): col for i,col in enumerate(rCols)}
            diff = {dCol: (df[lCol] - df[rCols[dCol]]) for dCol, lCol in lCols.items()}
            ddf = pd.DataFrame(diff)
            
            if idColumn != False:
                ddf = pd.concat([ddf, df[idColumn]], axis=1, join_axes=[df.index])
            return ddf

        self._state = P4thPDF._apply(self.state, do,
                                    re.compile(leftRe),
                                    re.compile(rightRe),
                                    reRepl,
                                    self.idColumn if idColumn == None else idColumn)
        return self
    
    def statistics(self, trueRe, guessRe, commonRe=r'\1', idColumn=None):
        
        def do(df, trueRe, guessRe, commonRe, idColumn):

            def pickTP(row, dfGuess):
                return row * dfGuess[row.name]

            def pickTN(row, dfGuess):
                return (1-row) * (1-dfGuess[row.name])

            
            dfTrue = P4thPDF(df.filter(regex=trueRe)).rename(trueRe, commonRe).state
            dfGuess = P4thPDF(df.filter(regex=guessRe)).rename(guessRe, commonRe).state
            dfDiff = P4thPDF(df).diff(trueRe, guessRe, commonRe, idColumn=False).state
            res = {
                'P': dfTrue.transform(lambda v: 1.0 * (v == 1)).aggregate(['sum'], axis='index').loc['sum'],
                'N': dfTrue.transform(lambda v: 1.0 * (v == 0)).aggregate(['sum'], axis='index').loc['sum'],
                'FP': dfDiff.transform(lambda v: 1.0 * (v == -1)).aggregate(['sum'], axis='index').loc['sum'],
                'FN': dfDiff.transform(lambda v: 1.0 * (v == 1)).aggregate(['sum'], axis='index').loc['sum'],
                'TP': dfTrue.apply(pickTP, axis='index', dfGuess=dfGuess).transform(lambda v: 1.0 * (v == 1)).aggregate(['sum'], axis='index').loc['sum'],
                'TN': dfTrue.apply(pickTN, axis='index', dfGuess=dfGuess).transform(lambda v: 1.0 * (v == 1)).aggregate(['sum'], axis='index').loc['sum'],

                }
            res.update({
                'Precision': res['TP'] / (res['TP'] + res['FP']),
                'Recall': res['TP'] / (res['TP'] + res['FN'])
            })
            res.update({
                'F1': 2 * (res['Precision'] * res['Recall']) / (res['Precision'] + res['Recall']),
            })
            res = pd.DataFrame.from_dict(res, orient='index')
            #print(res['Precision'])
            #print(res['Recall'])
            #print(res['F1'])
            #tmp = pd.DataFrame(tmp, index=tmp.keys())
            #print(res)
            #print((dfGuess==1))
            return res
            

        self._state = P4thPDF._apply(self.state, do,
                                    re.compile(trueRe),
                                    re.compile(guessRe),
                                    commonRe,
                                    self.idColumn if idColumn == None else idColumn)
        return self
    
    def reduce(self, F, initialValue):

        def do(df, F, initialValue):
            res = initialValue
            for index, row in df.iterrows():
                #print(row)
                res = F(res, row)
            return res

        self._state = P4thPDF._apply(self.state, do, F, initialValue)
        return self

    @property
    def idColumn(self):
        return P4thPDF._idColumn 

    @idColumn.setter
    def idColumn(self, name):
        P4thPDF._idColumn = name 
                            
    @property
    def state(self):
        return self._state
                            
    @staticmethod
    def _apply(state, do, *args, **kwargs):
        if isinstance(state, (tuple)):
            return tuple([P4thPDF._apply(state, do, *args, **kwargs) for state in list(state)])
        if isinstance(state, (list)):
            return [P4thPDF._apply(state, do, *args, **kwargs) for state in state]
        if isinstance(state, (dict)):
            return {key: P4thPDF._apply(state, do, *args, **kwargs) for key,state in state.items()}
        return do(state, *args, **kwargs)
        


if __name__ == "__main__":

    import math

    
    #catsAsDummies
    df = pd.DataFrame([
        [-3, 'A', 'odd'],
        [-2, 'B', 'even'],
        [-1, 'C', 'odd'],
        [0, 'A', 'even'],
        [1, 'B', 'odd'],
    ], columns=['lin', 'letter', 'group'])
    ddf = P4thPDF(df).catsAsDummies('letter').state
    #print(ddf)
    assert list(ddf['A']) == [True, False, False, True, False] 
    assert list(ddf['B']) == [False, True, False, False, True] 
    assert list(ddf['C']) == [False, False, True, False, False] 

    ddf = P4thPDF(df).catsAsDummies('letter', cats=['A','C','D'], reNamePattern=r'(A|B|C|D)', reNameReplace=r'_\1_', join=1, typer=int).state
    #print(ddf)
    assert list(ddf['lin']) == [-3, -2, -1, 0, 1] 
    assert list(ddf['_A_']) == [1, 0, 0, 1, 0] 
    assert list(ddf['_C_']) == [0, 0, 1, 0, 0] 
    assert list(ddf['_D_']) == [0, 0, 0, 0, 0] 

    #join
    ldf = pd.DataFrame([
        [-3, 9, 'odd'],
        [-2, 4, 'even'],
        [-1, 1, 'odd'],
        [0, 0, 'even'],
        [1, 1, 'odd'],
    ], columns=['lin', 'sqr', 'group'])
    rdf = pd.DataFrame([
        [-1, -1, 'odd'],
        [0, 0, 'even'],
        [1, 1, 'odd'],
        [2, 8,'even'],
        [3, 27,'odd'],
    ], columns=['lin', 'cbe', 'group'])

    jdf = P4thPDF(ldf).join(rdf, joinType='inner', on='lin', reRightColFilter='^cbe$').state
    #print(jdf)
    assert list(jdf['lin']) == [-1, 0, 1] 
    assert list(jdf['sqr']) == [1, 0, 1] 
    assert list(jdf['cbe']) == [-1, 0, 1] 
    assert list(jdf['group']) == ['odd', 'even', 'odd'] 

    jdf = P4thPDF(ldf).join(rdf, joinType='left', on=['lin'], reRightColFilter='.*', rightSuffix='-R').state
    #print(jdf)
    assert list(jdf['lin']) == [-3,-2,-1, 0, 1] 
    assert list(jdf['sqr']) == [9,4,1, 0, 1] 
    assert list(jdf['cbe-R'])[2:] == [ -1, 0, 1] 
    assert list(jdf['group']) == ['odd', 'even', 'odd', 'even', 'odd'] 
    assert list(jdf['group-R'])[2:] == ['odd', 'even', 'odd'] 
    
    #transformColumns
    df = pd.DataFrame([
        [-3, 9, 'odd'],
        [-2, 4, 'even'],
        [-1, 1, 'odd'],
        [0, 0, 'even'],
        [1, 1, 'odd'],
        [2, 4,'even'],
        [3, 9,'odd'],
    ], columns=['lin', 'sqr', 'group'])
    sdf = P4thPDF(df).transformColumns({
        '^lin$': str,
        'sqr': str,
        'dummy': str,
        'group': lambda val : 1 if val == 'odd' else 0
    }).state
    assert list(sdf['lin'].values) == ['-3', '-2', '-1', '0', '1', '2', '3']
    assert list(sdf['sqr'].values) == ['9', '4', '1', '0', '1', '4', '9']
    assert list(sdf['group'].values) == [1, 0, 1, 0, 1, 0, 1]

    #sortRows
    df = pd.DataFrame([
        [2, 4,'even'],
        [0, 0, 'even'],
        [-2, 4, 'even'],
        [-3, 9, 'odd'],
        [-1, 2, 'odd'],
        [1, 2, 'odd'],
        [3, 9,'odd'],
    ], columns=['lin', 'sqr', 'group'])
    sdf = P4thPDF(df).sortRows(by='lin').state
    assert list(sdf['lin'].values) == [-3, -2, -1, 0, 1, 2, 3]
    sdf = P4thPDF(df).sortRows(by=['group', 'sqr', 'lin']).state
    #print(sdf)
    assert list(sdf['lin'].values) == [0, -2, 2, -1, 1, -3, 3]
    #assert 1 == 0

    #rolling
    df = pd.DataFrame([
        [-3, 9, 'odd'],
        [-2, 4, 'even'],
        [-1, 2, 'odd'],
        [0, 0, 'even'],
        [1, 2, 'odd'],
        [2, 4,'even'],
        [3, 9,'odd'],
    ], columns=['lin', 'sqr', 'group'])
    sdf = P4thPDF(df).rolling(reFilter='lin|sqr', window=3, center=True, min_periods=1).state
    #print(sdf)
    assert list(sdf['lin'].values) == [-2.5, -2.0, -1.0, 0.0, 1.0, 2.0, 2.5]
    assert list(sdf['sqr'].values) == [(9+4)/2, (9+4+2)/3, (4+2+0)/3, (2+0+2)/3, (0+2+4)/3, (2+4+9)/3, (4+9)/2]

    sdf = P4thPDF(df).rolling(reFilter='lin', window=3, center=False, min_periods=1).state
    #print(sdf)
    assert np.isclose(list(sdf['lin'].values), [(-3)/1, (-2-3)/2, (-1-2-3)/3, (0-1-2)/3, (1-0-1)/3, (2+1+0)/3, (3+2+1)/3]).all()


    sdf = P4thPDF(df).rolling(reFilter='lin|sqr', groupBy='group',window=3, center=True, min_periods=1, prefix='<', suffix='>').state
    #print(sdf)
    assert list(sdf['<sqr>'].values) == [(9+2)/2, (4+0)/2, (9+2+2)/3, (4+0+4)/3, (2+2+9)/3, (0+4)/2, (2+9)/2]

    #lagged
    df = pd.DataFrame([
        [-3, 9, 'odd'],
        [-2, 4, 'even'],
        [-1, 1, 'odd'],
        [0, 0, 'even'],
        [1, 1, 'odd'],
        [2, 4,'even'],
        [3, 9,'odd'],
    ], columns=['lin', 'sqr', 'group'])
    sdf = P4thPDF(df).lagged(reFilter='sqr', groupBy='group', lag=1).state
    #print(sdf)
    assert list(sdf['lin'].values) == [-3, -2, -1, 0, 1, 2, 3]
    assert list(sdf['sqr'].values)[:-2] == [1.0, 0.0, 1.0, 4.0, 9.0]
    assert math.isnan(list(sdf['sqr'].values)[-2])
    assert math.isnan(list(sdf['sqr'].values)[-1])

    sdf = P4thPDF(df).lagged(reFilter='sqr', groupBy='group', lag=-1, prefix='<', suffix='>').state
    #print('sdf', sdf)
    assert list(sdf['lin'].values) == [-3, -2, -1, 0, 1, 2, 3]
    assert math.isnan(list(sdf['<sqr>'].values)[0])
    assert math.isnan(list(sdf['<sqr>'].values)[1])
    assert list(sdf['<sqr>'].values)[2:] == [9, 4, 1, 0, 1]

    #addColumns
    df = pd.DataFrame([
        [-2, 4],
        [-1, 2],
        [0, 0],
        [1, 2,],
        [2, 4,],
    ], columns=['lin', 'sqr'])
    sdf = P4thPDF(df).addColumns({
        'sum': lambda row : row['lin'] + row['sqr']
    }).state

    assert sdf['sum'][0] == 2
    assert sdf['sum'][1] == 1
    assert sdf['sum'][2] == 0
    assert sdf['sum'][3] == 3
    assert sdf['sum'][4] == 6
    
    #group
    df = pd.DataFrame([
        ['A', 1, 10, 100,],
        ['A', 2, 20, 400,],
        ['A', 1, 30, 900,],
        ['A', 2, 40, 1600,],
        ['A', 2, 20, 400,],
        ['B', 1, 1, 1,],
        ['B', 2, 2, 4,],
        ['B', 1, 3, 9,],
        ['B', 2, 4, 16,],
    ], columns=['TYPE', 'AGE', 'VALUE', 'VV'])
    gdf = P4thPDF(df).group(['TYPE', 'AGE'], {'_Min': np.min, '_Med': np.median, '_Max': np.max}, 'LEN').state
    #print(gdf)
    assert 'VV_Med' in gdf
    assert 'VV_Max' in gdf
    a1 = gdf[(gdf['TYPE'] == 'A') & (gdf['AGE'] == 1)].reset_index()
    assert a1['VALUE_Min'][0] == 10
    assert a1['VALUE_Med'][0] == 20
    assert a1['VALUE_Max'][0] == 30
    assert a1['LEN'][0] == 2

    a2 = gdf[(gdf['TYPE'] == 'A') & (gdf['AGE'] == 2)].reset_index()
    assert a2['VALUE_Min'][0] == 20
    assert a2['VALUE_Med'][0] == 20
    assert a2['VALUE_Max'][0] == 40
    assert a2['LEN'][0] == 3

    b2 = gdf[(gdf['TYPE'] == 'B') & (gdf['AGE'] == 2)].reset_index()
    assert b2['VALUE_Min'][0] == 2
    assert b2['VALUE_Med'][0] == 3
    assert b2['VALUE_Max'][0] == 4
    assert b2['LEN'][0] == 2

    b1 = gdf[(gdf['TYPE'] == 'B') & (gdf['AGE'] == 1)].reset_index()
    assert b1['VV_Min'][0] == 1
    assert b1['VV_Med'][0] == 5
    assert b1['VV_Max'][0] == 9
    assert b1['LEN'][0] == 2

    gdf = P4thPDF(df).group(['TYPE', 'AGE'], {'_Min': np.min, '_Mean': np.mean, '_Max': np.max}, countLabel='LEN', joinType='left', reColumnFilter=r'^VALUE$').state
    print(gdf)
    assert 'VV_Mean' not in gdf
    assert 'VV_Max' not in gdf
    assert list(gdf['AGE']) == [1,2,1,2,2,1,2,1,2]
    assert list(gdf['LEN']) == [2,3,2,3,3,2,2,2,2]
    assert list(gdf['VALUE_Mean']) == [(10+30)/2,(20+40+20)/3,(10+30)/2,(20+40+20)/3,(20+40+20)/3, (1+3)/2,(2+4)/2, (1+3)/2,(2+4)/2]

    #columns
    df = pd.DataFrame([
        [1, 10,],
        [2, 20,],
        [3, 30,],
        [4, 40,],
        [5, 50,],
        [6, 60,],
    ], columns=['A', 'BA'])
    Bs = P4thPDF(df).columns('^B').state
    As = P4thPDF(df).columns('.*A$').state
    assert Bs == ['BA']
    assert As == ['A', 'BA']

    #partitionRows
    df = pd.DataFrame([
        [1, 10,],
        [2, 20,],
        [3, 30,],
        [4, 40,],
        [5, 50,],
        [6, 60,],
    ], columns=['A', 'AA'])
    
    dfA = P4thPDF(df).partitionRows(lambda row, i: 'even' if row['A'] % 2 == 0 else 'odd').state
    assert dfA['even']['A'][0] == 2
    assert dfA['even']['AA'][0] == 20
    assert dfA['even']['A'][1] == 4
    assert dfA['even']['AA'][1] == 40
    assert dfA['even']['A'][2] == 6
    assert dfA['even']['AA'][2] == 60

    assert dfA['odd']['A'][0] == 1
    assert dfA['odd']['AA'][0] == 10
    assert dfA['odd']['A'][1] == 3
    assert dfA['odd']['AA'][1] == 30
    assert dfA['odd']['A'][2] == 5
    assert dfA['odd']['AA'][2] == 50
    
    #reduce
    df = pd.DataFrame([
        [1, 2,],
        [3, 4,],
    ], columns=['A', 'B'])

    s = P4thPDF(df) \
          .reduce(lambda res, row: res + row['A'] + row['B'], 0) \
          .state
    #print(s)
    assert s == 10

    m = P4thPDF(df) \
          .reduce(lambda res, row: res * row['A'] * row['B'], 1) \
          .state
    #print(m)
    assert m == 24

    #ensureColumns
    df = pd.DataFrame([
        [1, 3,],
        [1, 3,],
    ], columns=['D1', 'D3'])
    df = P4thPDF(df).ensureColumns({'D1': 100, 'D2': 2, 'D0': 0}).state
    #print(df)
    assert df.D0[0] == 0
    assert df.D0[1] == 0

    assert df.D1[0] == 1
    assert df.D1[1] == 1

    assert df.D2[0] == 2
    assert df.D2[1] == 2

    assert df.D3[0] == 3
    assert df.D3[1] == 3



    # dummiesAsOneHot
    df = pd.DataFrame([
        [1, 1, 1,], #0 <-- will be removed!
        [0, 0, 0,], #1
        [1, 0, 0,], #2
        [0, 1, 0,], #3
        [0, 0, 1,], #4
        [1, 1, 0,], #5
        [1, 1, 1,], #6
        [0, 0, 0,], #7
    ], columns=['d_A', 'd_B', 'd_C'])
    df = df[1:]
    df = P4thPDF(df).identify().state
    #print(df)
    hdf = P4thPDF(df).dummiesAsOneHot(r'(d)(_)([A-Z])', 'NONE', typer=int, join=True).state
    #print(hdf)
    assert hdf.shape[0] == 7
    assert hdf.shape[1] == 7

    assert hdf.d_NONE[1] == 1
    assert hdf.d_NONE[2] == 0
    assert hdf.d_NONE[3] == 0
    assert hdf.d_NONE[4] == 0
    assert hdf.d_NONE[5] == 0
    assert hdf.d_NONE[6] == 0
    assert hdf.d_NONE[7] == 1

    assert hdf.d_A_B_C[1] == 0
    assert hdf.d_A_B_C[2] == 0
    assert hdf.d_A_B_C[3] == 0
    assert hdf.d_A_B_C[4] == 0
    assert hdf.d_A_B_C[5] == 0
    assert hdf.d_A_B_C[6] == 1
    assert hdf.d_A_B_C[7] == 0

    assert hdf.d_A[1] == 0
    assert hdf.d_A[2] == 1
    assert hdf.d_A[3] == 0
    assert hdf.d_A[4] == 0
    assert hdf.d_A[5] == 0
    assert hdf.d_A[6] == 0
    assert hdf.d_A[7] == 0

    assert hdf.d_B[1] == 0
    assert hdf.d_B[2] == 0
    assert hdf.d_B[3] == 1
    assert hdf.d_B[4] == 0
    assert hdf.d_B[5] == 0
    assert hdf.d_B[6] == 0
    assert hdf.d_B[7] == 0

    assert hdf.d_C[1] == 0
    assert hdf.d_C[2] == 0
    assert hdf.d_C[3] == 0
    assert hdf.d_C[4] == 1
    assert hdf.d_C[5] == 0
    assert hdf.d_C[6] == 0
    assert hdf.d_C[7] == 0

    #keepColumns
    df = pd.DataFrame([
        [0, 0, 1, 1, 0, 0,],
        [1, 0, 0, 0, 0, 1,],
        [0, 1, 0, 1, 0, 0,],
    ], columns=['D2', 'D3', 'D1', 'ML1', 'ML2', 'ML3'])
    df = P4thPDF(df).keepColumns(r'^D.*').state
    #print(df)
    assert df.D2[0] == 0
    assert df.D2[1] == 1
    assert df.D2[2] == 0

    assert df.D3[0] == 0
    assert df.D3[1] == 0
    assert df.D3[2] == 1

    assert df.D1[0] == 1
    assert df.D1[1] == 0
    assert df.D1[2] == 0

    assert('ML1' not in df)
    assert('ML2' not in df)
    assert('ML3' not in df)
    
    #hasColumns
    df = pd.DataFrame([
        [0, 0, 1, 1, 0, 0,],
        [1, 0, 0, 0, 0, 1,],
        [0, 1, 0, 1, 0, 0,],
    ], columns=['D2', 'D3', 'D1', 'ML1', 'ML2', 'ML3'])
    assert P4thPDF(df).hasColumns(r'^D.*').state
    assert not P4thPDF(df).hasColumns(r'^D4.*').state
    assert not P4thPDF(df).hasColumns(r'^Q.*').state
    

    #keepRow
    df = pd.DataFrame([
        [1, 0, 0,],
        [1, 1, 0,],
        [1, 1, 1,],
    ], columns=['D1', 'D2', 'D3'])
    df0 = P4thPDF(df).keepRows(lambda row: row.sum() > 3).state
    #print(df0)
    assert df0.shape[0] == 0

    df1 = P4thPDF(df).keepRows(lambda row: row.sum() > 2).state
    #print(df1)
    assert df1.shape[0] == 1
    assert df1.D1[0] == 1
    assert df1.D2[0] == 1
    assert df1.D3[0] == 1

    df2 = P4thPDF(df).keepRows(lambda row: row.sum() > 1).state
    #print(df2)
    assert df2.shape[0] == 2
    assert df2.D1[0] == 1
    assert df2.D2[0] == 1
    assert df2.D3[0] == 0
    assert df2.D1[1] == 1
    assert df2.D2[1] == 1
    assert df2.D3[1] == 1

    df3 = P4thPDF(df).keepRows(lambda row: row.sum() > 0).state
    #print(df3)
    assert df3.shape[0] == 3
    assert df3.D1[0] == 1
    assert df3.D2[0] == 0
    assert df3.D3[0] == 0
    assert df3.D1[1] == 1
    assert df3.D2[1] == 1
    assert df3.D3[1] == 0
    assert df3.D1[2] == 1
    assert df3.D2[2] == 1
    assert df3.D3[2] == 1

    # Diff
    df = pd.DataFrame([
        [0, 0, 1, 1, 0, 0,],
        [1, 0, 0, 0, 0, 1,],
        [0, 1, 0, 1, 0, 0,],
    ], columns=['D2', 'D3', 'D1', 'ML1', 'ML2', 'ML3'])
    df = P4thPDF(df).identify().diff(r'^D(.*)$', r'^ML(.*)$').state

    assert df.diff_1[0] == 0
    assert df.diff_2[0] == 0
    assert df.diff_3[0] == 0

    assert df.diff_1[1] == 0
    assert df.diff_2[1] == 1
    assert df.diff_3[1] == -1

    assert df.diff_1[2] == -1
    assert df.diff_2[2] == 0
    assert df.diff_3[2] == 1

    # Transform
    df = pd.DataFrame([
        [0.7, 45],
        [0.5, 46],
        [0.3, 23],
    ], columns=['P1', 'X'])
    df = P4thPDF(df).transform(lambda x: x**2).state
    assert df.P1[0] == 0.7**2
    assert df.P1[1] == 0.5**2
    assert df.P1[2] == 0.3**2
    assert df.X[0] == 45**2
    assert df.X[1] == 46**2
    assert df.X[2] == 23**2

    # CollpaseDummies
    df = pd.DataFrame([
        [1, 0, 'a'],
        [0, 1, 'b'],
        [1, 1, 'c'],
    ], columns=['d1', 'd2', 'c'])
    cdf = P4thPDF(df).collapseDummies(r'^d[12]$', column='d').state
    #print(cdf)
    assert 'd1' not in cdf
    assert 'd2' not in cdf
    assert 'c' not in cdf
    assert 'd' in cdf
    assert cdf.d[0] == ['d1']
    assert cdf.d[1] == ['d2']
    assert cdf.d[2] == ['d1', 'd2']
    
    cdf = P4thPDF(df).collapseDummies(r'^d[12]$', column='d', join=True).state
    assert 'd1' in cdf
    assert 'd2' in cdf
    assert 'c' in cdf
    assert 'd' in cdf

    assert cdf.d1.tolist() == df.d1.tolist()
    assert cdf.d2.tolist() == df.d2.tolist()
    assert cdf.c.tolist() == df.c.tolist()

    assert cdf.d[0] == ['d1']
    assert cdf.d[1] == ['d2']
    assert cdf.d[2] == ['d1', 'd2']
    
    idf = P4thPDF(df).identify('iidd').state
    cdf = P4thPDF(idf).collapseDummies(r'^d[12]$', column='d', join=True, idColumn='iidd').state
    #print(cdf)
    assert 'd1' in cdf
    assert 'd2' in cdf
    assert 'c' in cdf
    assert 'd' in cdf

    assert cdf.d1.tolist() == idf.d1.tolist()
    assert cdf.d2.tolist() == idf.d2.tolist()
    assert cdf.c.tolist() == idf.c.tolist()
    assert cdf['iidd'].tolist() == idf['iidd'].tolist()


    assert cdf.d[0] == ['d1']
    assert cdf.d[1] == ['d2']
    assert cdf.d[2] == ['d1', 'd2']
    
    idf = P4thPDF(df).identify().state
    cdf = P4thPDF(idf).collapseDummies(r'^d[12]$', column='d', join=True, idColumn=None).state
    #print(cdf)
    assert 'd1' in cdf
    assert 'd2' in cdf
    assert 'c' in cdf
    assert 'd' in cdf

    assert cdf.d1.tolist() == idf.d1.tolist()
    assert cdf.d2.tolist() == idf.d2.tolist()
    assert cdf.c.tolist() == idf.c.tolist()
    assert cdf[P4thPDF._idColumn].tolist() == idf[P4thPDF._idColumn].tolist()


    assert cdf.d[0] == ['d1']
    assert cdf.d[1] == ['d2']
    assert cdf.d[2] == ['d1', 'd2']
    
    # Most Likely
    df = pd.DataFrame([
        [0.7, 45, 0.03, 0.4],
        [0.5, 46, 0.9, 0.7],
        [0.3, 23, 0.4, 0.9],
    ], columns=['P1', 'X', 'P2', 'P3'])
    df = P4thPDF(df).mostLikely(r'^P(.*)', r'ML\1', idColumn=False).state
    assert df.ML1[0] == 1
    assert df.ML2[0] == 0
    assert df.ML3[0] == 0

    assert df.ML1[1] == 0
    assert df.ML2[1] == 1
    assert df.ML3[1] == 0

    assert df.ML1[2] == 0
    assert df.ML2[2] == 0
    assert df.ML3[2] == 1


    # Statistics
    df = pd.DataFrame([
        [1, 1, 1, 0, 0, 0,],
        [1, 1, 1, 0, 0, 1,],
        [1, 1, 1, 0, 1, 0,],
        [1, 1, 1, 0, 1, 1,],
        [1, 1, 1, 0, 0, 0,],
        [1, 1, 1, 0, 0, 1,],
        [1, 1, 1, 0, 1, 0,],
        [1, 1, 1, 1, 1, 1,],
    ], columns=['D1', 'ML1', 'D2', 'ML2', 'D3', 'ML3'])
    df = P4thPDF(df).identify().statistics(r'^D(.*)$', r'^ML(.*)$', r'D\1').state
    assert(df.loc['P']['D1'] == 8)
    assert(df.loc['P']['D2'] == 8)
    assert(df.loc['P']['D3'] == 4)

    assert(df.loc['TP']['D1'] == 8)
    assert(df.loc['TP']['D2'] == 1)
    assert(df.loc['TP']['D3'] == 2)

    assert(df.loc['FN']['D1'] == 0)
    assert(df.loc['FN']['D2'] == 7)
    assert(df.loc['FN']['D3'] == 2)

    assert(df.loc['Precision']['D1'] == 8 / (8 + 0))
    assert(df.loc['Precision']['D2'] == 1 / (1 + 0))
    assert(df.loc['Precision']['D3'] == 2 / (2 + 2))

    assert(df.loc['Recall']['D1'] == 8 / (8 + 0))
    assert(df.loc['Recall']['D2'] == 1 / (1 + 7))
    assert(df.loc['Recall']['D3'] == 2 / (2 + 2))

    assert(df.loc['F1']['D1'] == 2 * (1 * 1) / (1 + 1))
    assert(df.loc['F1']['D2'] == 2 * (1 * (1/8)) / (1 + (1/8)))
    assert(df.loc['F1']['D3'] == 2 * ((2/4) * (2/4)) / ((2/4) + (2/4)))
