'''Create attributes from family tree data
'''

import os
import os.path
from subprocess import check_call
import cPickle as pickle

import numpy as np

import data; reload(data)
from data import TitanicDataSet
from findfamilies import construct_family_components, child_parent_direction


def main():
    train = TitanicDataSet.get_train()
    test = TitanicDataSet.get_test()
    families = construct_family_components(train, test)
    people = [mark_problems(p, f) for f in families for p in f.nodes]
    synthesize('train', [p for p in people if p.survived is not None], train)
    synthesize('test', [p for p in people if p.survived is None], test)

def mark_problems(p, f):
    p.difficult_parent_child = f.difficult_parent_child
    return p

def synthesize(name, people, original_ds):
    # Correct order or individuals
    assert len(people) == len(original_ds)
    name_orders = list(original_ds.name)
    people = sorted(people, key=lambda p: name_orders.index(p.a.name))
    assert all(p.a.name == name for p,name in zip(people, original_ds.name))

    base_keys = people[0].a._fields
    synthesized_keys, calculates = zip(*synthesized_attributes)
    keys = base_keys + synthesized_keys
    rows = [map(coere_attribute, p.a) +
            [c(p) for c in calculates] for p in people]
    cols = map(np.array, zip(*rows))
    global ds
    ds = TitanicDataSet(keys, cols, people[0].survived is not None)
    with open('data/synthesized/%s.p' % (name,), 'w') as fp:
        pickle.dump(ds, fp, 2)

def coere_attribute(v):
    return v

def calculate_spouse_survived(p):
    if p.spouse is not None and p.spouse.survived is not None:
        return int(p.spouse.survived)
    return -1

def iter_children(p):
    if not p.difficult_parent_child:
        for c in p.children:
            yield c
        return
    for e in p.edges:
        o = e.other(p)
        if e.definitive_child and child_parent_direction(p, o):
            yield o

def iter_parents(p):
    if not p.difficult_parent_child:
        for pp in p.known_parents:
            yield pp
        return
    for e in p.edges:
        o = e.other(p)
        if e.definitive_child and child_parent_direction(o, p):
            yield o

def iter_siblings(p):
    if not p.difficult_parent_child:
        for s in p.siblings:
            yield s
        return
    for e in p.edges:
        if e.definitive_sibling:
            yield e.other(p)

def iter_extended(p):
    if not p.difficult_parent_child:
        for e in p.extendeds:
            yield e
        return
    for e in p.edges:
        if e.definitive_extended:
            yield e.other(p)

def make_count(itr):
    return lambda p: sum(1 for o in itr(p))

def make_count_survived(itr):
    return lambda p: sum(1 for o in itr(p) if o.survived is True)

def make_count_died(itr):
    return lambda p: sum(1 for o in itr(p) if o.survived is False)


synthesized_attributes = [
    ('title', lambda p: p.parsed_name.title),
    ('had_nickname', lambda p: bool(p.parsed_name.nick)),
    ('had_othername', lambda p: bool(p.parsed_name.other)),

    ('had_spouse', lambda p: p.spouse is not None),
    ('spouse_survived', calculate_spouse_survived),

    ('n_children', make_count(iter_children)),
    ('n_children_died', make_count_died(iter_children)),
    ('n_children_survived', make_count_survived(iter_children)),
#    ('n_children_unkown', calculate_n_children_unkown),
    ('n_parents', make_count(iter_parents)),
    ('n_parents_died', make_count_died(iter_parents)),
    ('n_parents_survived', make_count_survived(iter_parents)),
#    ('n_parents_unkown', calculate_n_parents_unkown),

    ('n_sibling', make_count(iter_siblings)),
    ('n_sibling_died', make_count_died(iter_siblings)),
    ('n_sibling_survived', make_count_survived(iter_siblings)),

    ('n_extended', make_count(iter_extended)),
    ('n_extended_died', make_count_died(iter_extended)),
    ('n_extended_survived', make_count_survived(iter_extended)),
]


__name__ == '__main__' and main()
