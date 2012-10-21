'''Provides a convenient interface to the training and testing data
'''

from __future__ import division

import os.path
import csv
from functools import wraps
from collections import namedtuple

import numpy as np

csv_dir = os.path.join(os.path.dirname(__file__), 'data/csv')
train_path = os.path.join(csv_dir, 'train.csv')
test_path = os.path.join(csv_dir, 'test.csv')

attribute_types = dict(
  survived='bool',
  pclass=('nominal', ('1','2','3')),
  name='string',
  sex=('nominal', ('male','female')),
  age='float',
  sibsp='int',
  parch='int',
  ticket='string',
  fare='float',
  cabin='string',
  embarked=('nominal', ('C','S','Q')),
  )

def process_bool(values, args):
    mp = {'0':False, '1':True}
    return np.array([mp[v] for v in values], dtype=np.bool)

def process_nominal(values, args):
    acc = []
    for v in values:
        if not v:
            acc.append(-1)
        elif v in args:
            acc.append(args.index(v))
        else:
            raise ValueError('bad value %r, not in %s' % (v, args))
    return np.array(acc, dtype=np.int)

def process_int(values, args):
    return process_number(values, int)

def process_float(values, args):
    return process_number(values, float)

def process_number(values, tp):
    acc = []
    for v in values:
        if not v:
            acc.append(-1)
        else:
            v = tp(v)
            if v < 0:
                raise ValueError('negative number')
            acc.append(v)
    return np.array(acc, dtype=tp)

def process_float(values, args):
    return np.array([float(v) if v else -1.0 for v in values], dtype=np.double)

def process_string(values, args):
    return np.array(values)


attribute_types_processors=dict(
    bool=process_bool,
    nominal=process_nominal,
    int=process_int,
    float=process_float,
    string=process_string)


def memorize(func):
    @wraps(func)
    def wrapper(*args):
        try:
            return cache[args]
        except KeyError:
            result = cache[args] = func(*args)
            return result
    wrapper.cache = cache = {}
    return wrapper

@memorize
def get_entry_class(keys):
    return namedtuple('Entry', keys)

class TitanicDataSet(object):

    def __init__(self, keys, columns, is_train):
        assert len(keys) == len(columns)
        assert len(set(map(len, columns))) == 1
        self.keys = keys
        self.columns = columns
        self.is_train = is_train
        self.entry_class = get_entry_class(keys)

    def __reduce__(self):
        return (self.__class__, (self.keys, self.columns, self.is_train))

    def __len__(self):
        return len(self.columns[0])

    def get_column(self, key, copy=True):
        try:
            i = self.keys.index(key)
        except ValueError:
            raise ValueError("no such column %r" % (key,))
        return self.columns[i].copy() if copy else self.columns[i]

    def __getattr__(self, name):
        if name in self.keys:
            return self.get_column(name, copy=False)
        raise AttributeError(name)

    def get_attributes(self, *keys):
        return np.array([self.get_column(key) for key in keys],
                        copy=True,
                        dtype=float).T

    def splice(self, mask):
        return self.__class__(self.keys,
                              [c[mask] for c in self.columns],
                              self.is_train)

    def get_entry(self, i):
        return self.entry_class(*(c[i] for c in self.columns))

    def iter_entries(self):
        for i in xrange(len(self)):
            yield self.get_entry(i)

    def copy(self):
        return self.__class__(self.keys,
                              [c.copy() for c in self.columns],
                              self.is_train)

    @classmethod
    @memorize
    def get_train(cls):
        return cls.load_train().copy()

    @classmethod
    @memorize
    def get_test(cls):
        return cls.load_test().copy()

    @classmethod
    def load_train(cls):
        return cls.load(train_path, True)

    @classmethod
    def load_test(cls):
        return cls.load(test_path, False)

    @classmethod
    def load(cls, path, is_train):
        with open(path) as fp:
            reader = csv.reader(fp)
            rows = list(reader)
        keys = tuple(rows.pop(0))
        columns = zip(*rows)
        columns = [cls.process_column(key, column) for key,column in zip(keys, columns)]
        return cls(keys, columns, is_train)

    @classmethod
    def process_column(cls, key, column):
        tp = attribute_types[key]
        if isinstance(tp, tuple):
            tp,args = tp
        else:
            args = ()
        return attribute_types_processors[tp](column, args)


