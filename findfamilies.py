'''Hueristic-based algorithim to find the family trees in the titanic data.


'''
from __future__ import division

import re
from contextlib import contextmanager

import numpy as np

from data import TitanicDataSet
import graphlib

# Parameters
#--------------------------------------------------------
MINIMUM_AGE_FOR_MARRIAGE = 14
# At most, a married woman can be n years older than husband
# This rule might not be needed
LARGEST_MARRIED_FEMALE_AGE_ADVANTAGE = 10
# A parent must be atleast n years older than their child
MINIMUM_PARENT_AGE_ADVANTAGE = 14


# Graph data structures
#--------------------------------------------------------

class DotIDMixin(object):
    '''Base class for nodes that are writen to dot files
    '''

    dot_id_counter = 0
    def __init__(self):
        self.dot_id = str(self.__class__.dot_id_counter)
        self.__class__.dot_id_counter += 1


name_rgx = re.compile(r'''
([^,]+) # Last Name
, \s+
  ([^.]+) \. # Title
  \s+
  ([^("]+)? # Main name
  (?:
    "([^"]+)" # Nick name
  )?
  (?:
    \( "?
      ([^)"]+) # Other name
    "? \)
  )?
''', re.VERBOSE)

class ParsedName(object):

    def __init__(self, last, title, main, nick, other):
        if not main and other:
            main = other.split(None, 0)
        self.last = last
        self.title = title
        self.main = main
        self.nick = nick
        self.other = other

    @classmethod
    def create(cls, name):
        m = name_rgx.match(name.lower())
        if not m:
            raise ValueError('bad name %r' % (name,))
        return cls(*m.groups())

    def iter_last_names(self):
        yield self.last
        if self.other:
            yield self.other.rsplit(None, 1)[-1]


class Person(DotIDMixin, graphlib.Node):

    write_elsewhere = False

    def __init__(self, attributes, survived):
        super(Person, self).__init__()
        self.survived = survived
        self.a = attributes
        self.parsed_name = ParsedName.create(attributes.name)
        # These are filled in as relationships are proven
        self.spouse = None
        self.mother = None
        self.father = None
        self.children = ()
        self.siblings = ()
        self.extended = ()

    @property
    def known_parents(self):
        a = []
        if self.mother: a.append(self.mother)
        if self.father: a.append(self.father)
        return a

    @property
    def n_known_parents(self):
        s = 0
        if self.mother: s+=1
        if self.father: s+=1
        return s

    @property
    def n_known_children(self):
        return len(self.children)

    @property
    def n_known_siblings(self):
        return len(self.siblings)

    @property
    def n_kownn_extended(self):
        return len(self.extended)

    def __str__(self):
        return 'p(%s)' % (self.name,)

    @property
    def adjusted_sibsp(self):
        return self.sibsp - len(self.siblings) - (1 if self.spouse else 0)

    @property
    def adjusted_parch(self):
        return self.a.parch - self.n_known_parents - self.n_known_children

    def get_edge_to(self, other):
        es = [e for e in self.edges if e.other(self) == other]
        if len(es) != 1:
            raise ValueError("%d edges to other" % (len(es),))
        return es[0]

    def has_edge_to(self, other):
        return any(1 for e in self.edges if e.other(self) == other)


class CommonLastName(graphlib.Node):
    '''Keeps track of individuals who share a common last name
    '''

    def __init__(self, lastname):
        super(CommonLastName, self).__init__()
        self.lastname = lastname

class LastNameEdge(graphlib.Edge):

    def __init__(self, name, person):
        if not isinstance(name, CommonLastName):
            name,person = person,name
        assert isinstance(name, CommonLastName), 'name %r' % (name,)
        assert isinstance(person, Person), 'person %r' % (person,)
        super(LastNameEdge, self).__init__(name, person)
        self.name = name
        self.person = person


class RelationEdge(graphlib.Edge):
    '''Keeps track of what relationships are possible between
       two linked individuals.
    '''

    def __init__(self, a, b):
        assert isinstance(a, Person)
        assert isinstance(b, Person)
        super(RelationEdge, self).__init__(a, b)
        self.a = a
        self.b = b
        self.set_all_possbilities(True)

    def set_all_possbilities(self, value):
        value = bool(value
        self.could_be_spouse = value
        self.could_be_sibling = value
        self.could_be_child = value
        self.could_be_extended = value

    @property
    def possibilities(self):
        return (self.could_be_spouse,
                self.could_be_sibling,
                self.could_be_child,
                self.could_be_extended)

    @property
    def n_possibilities(self):
        return sum(self.possibilities)

    @property
    def is_definitive(self):
        return self.n_possibilities == 1

    @property
    def definitive_spouse(self):
        self.check_definitive(0)

    @property
    def definitive_sibling(self):
        self.check_definitive(1)

    @property
    def definitive_child(self):
        return self.check_definitive(2)

    @property
    def definitive_extended(self):
        return self.check_definitive(3)

    def check_definitive(self, i):
        return self.possibilities[i] and self.is_definitive

    @property
    def could_be_close(self):
        return (self.could_be_spouse or
                self.could_be_sibling or
                self.could_be_child)


class LastNameBuilder(graphlib.GraphBuilder):

    def __init__(self):
        super(LastNameBuilder, self).__init__(node_factory=CommonLastName,
                                          edge_factory=LastNameEdge)

    def add_edge(self, name, person):
        self.values_to_nodes[person] = person
        super(LastNameBuilder, self).add_edge(self.get_node(name),
                                          person)

class NuclearFamily(DotIDMixin):

    def __init__(self, name='', mother=None, father=None, children=()):
        super(NuclearFamily, self).__init__()
        self.name = name
        self.mother = mother
        self.father = father
        self.children = list(children)
        if None not in (mother, father):
            assert mother.spouse is father and father.spouse is mother
        # add more validation


def construct_familiy_components(train=TitanicDataSet.get_train(),
                                 test=TitanicDataSet.get_test()):
    lnb = LastNameBuilder()
    add_last_names(lnb, train)
    add_last_names(lnb, test)
    last_name_graph = lnb.get_graph()
    return [tune_family_relations(f)
            for c in last_name_graph.components
            for f in build_relations(c)]

def add_last_names(nb, ds):
    if ds is None:
        return
    if ds.is_train:
        survived = ds.survived
    else:
        survived = [None] * len(ds)
    for attributes,survivied in zip(survivied, ds.iter_entries()):
        person = Person(attributes, survivied)
        for last_name in person.parsed_name.iter_last_names():
            nb.add_edge(last_name, person)

def build_relations(c):
    nodes, edges = c.teardown()
    people = [n for n in nodes if isinstance(n, Person)]
    gb = graphlib.GraphBuilder(edge_factory=RelationEdge)
    for p in people:
        gb.values_to_nodes[p] = p
    for i,a in enumerate(people):
        for b in people[i+1::]:
            if share_name(a, b) and general_affinity(a, b) != 0:
                gb.add_edge(a, b)
    return gb.get_graph().components


def find_nuclear_families(c):
    families = []
    seen = set()
    for n in c.nodes:
        if n not in seen and n.children:
            families.append(build_family(seen, n))

    extra_nodes = []
    extra_edges = set()
    for n in c.nodes:
        if n in seen:
            continue
        extra_nodes.append(n)
        for e in n.edges:
            extra_edges.add(e)

    for f in families:
        for p in f.mother, f.father:
            if p and set(p.edges) & extra_edges:
                p.write_elsewhere = True

    return families, extra_nodes, extra_edges

def build_family(seen, n):
    f = NuclearFamily(name=n.parsed_name.last)
    if n.sex == 0:
        f.father = n
    else:
        f.mother = n
    seen.add(n)
    if n.spouse:
        if n.spouse.sex == 0:
            f.father = n.spouse
        else:
            f.mother = n.spouse
        seen.add(n.spouse)

    f.children = n.children
    for c in n.children:
        seen.add(c)
        if c.children:
            build_family(seen, c)
    return f

def tune_family_relations(c):
    # Use simple hueristics to determine relationship individuals.
    # Throughout this procedure we incrementally prove various relationships
    # exists.
    update_relationship_possibilities(c)

    prove_spouses(c)
    update_relationship_possibilities(c)
    # ensure there is no ambiguity in spouse classification
    for e in c.edges:
        assert (not e.could_be_spouse) or e.definitive_spouse

    prove_parents(c)
    update_relationship_possibilities(c)
    return c

def update_relationship_possibilities(c):
    for e in c.edges:
        update_a_relationship_possibilities(e)

def update_a_relationship_possibilities(e):
    # Relationship type has already been figured out, nothing to do
    if e.is_definitive:
        return

    e.could_be_spouse = could_be_spouse(e.a, e.b)
    e.could_be_sibling = could_be_sibling(e.a, e.b)

    # child rule is directional, so try both directions
    c = could_be_child(e.a, e.b)
    if not c:
        c = could_be_child(e.b, e.a)
        if c:
            e.b, e.a = e.a, e.b
    e.could_be_child = c

    # If no possibility for close relationships, then classify as an extended relationship
    e.could_be_extended = not e.could_be_close

    assert e.n_possibilities >= 1


# Relationship possibilities functions
#--------------------------------------------------------

def could_be_sibling(a, b):
    # already proven that they are spouses
    if a.spouse is not None and a.spouse is b.spouse:
        assert b.spouse is a.spouse
        return True
    # proven that they aren't spouses
    elif a.spouse or b.spouse:
        return False

    # check if sibsp or sex rules out possibility of them being spoues
    if a.adjusted_sibsp == 0 or b.adjusted_sibsp == 0:
        return False
    if a.a.sex == b.a.sex:
        return False
    # parch should only be off by at most 2, as a couple has the same
    # number of kids, and only their own parents can lead to a difference
    if abs(a.a.parch - b.a.parch) > 2:
        return False

    # check if their names are consistent with them being married
    if a.parsed_name.last != b.parsed_name.last:
        return False
    titles = a.parsed_name.title, b.parsed_name.title
    if 'mrs' not in titles:
        return False
    if 'master' in titles or 'miss' in titles:
        return False

    # check if the womens main name includes a significant portion of
    # the mans name. (e.g Mrs. Sammuel Herman and Mr. Sammuel Herman)
    m,f = (a,b) if a.a.sex == 0 else (b,a)
    if m.parsed_name.main == f.parsed_name.main:
        return True
    n = largest_common_substring(m.parsed_name.main, f.parsed_name.main)
    if n > 0.5 * len(f.parsed_name.main):
        return True

    # rule out individual under the age of 14
    if not (ambiguous_gt(a.age, MINIMUM_AGE_FOR_MARRIAGE) and
            ambiguous_gt(b.age, MINIMUM_AGE_FOR_MARRIAGE)):
        return False

    # Tf we know both ages, rule out women who are 5+ years older than men
    # this might be an invalid rule as some of entries look like older rich
    # women married to young men. The cases i've observed are already handled
    # by the earlier check if the womens main name includes the main.
    # This rule was added to help sort out the mother/son vs. mother/husband,
    # but it may be better to leave those cases ambiguous anyways.
    if a.age != b.age != -1:
        if f.age > m.age + LARGEST_MARRIED_FEMALE_AGE_ADVANTAGE:
            return 0

    # optimistic default
    return True

def could_be_child(parent, child):
    # check if we've already proven that parent or their spouse is
    # a parent to this child
    if set(child.known_parents()) & set(filter(None, [parent, parent.spouse])):
        return True
    # check if parch rules out possibility of parent/child relationship
    if parent.a.parch == 0 or child.a.parch == 0:
        return False
    # rule out possibility of parents conceiving a child when under
    # a certain age
    if not ambiguous_gt_diff(parent.age, child.age, MINIMUM_PARENT_AGE_ADVANTAGE):
        return False
    # ensure they have the same last name
    if parent.parsed_name.last != maiden_name(child):
        return False
    # this rule can help with sibling/child ambiguities if we've
    # already found out they share the same parent
    if has_common_parents(parent, child):
        return False
    # optimistic default
    return True

def score_sibling(a, b):
    # Already proben that they are siblings
    if a in b.siblings:
        assert b in a.siblings
        return True
    # Check if sibsp rules out the possibility of them being siblings.
    # Adjust sibsp if we've proven they have a spouse.
    if a.adjusted_sibsp == 0 or b.adjusted_sibsp == 0:
        return False
    # if one of this individuals is a married women, then her maiden
    # name must match the other individuals last name to be siblings
    if maiden_name(a) != maiden_name(b):
        return False
    # this rule can help with sibling/child ambiguities if we've
    # already found out they share the same parent
    if has_common_parents(a, b):
        return True
    # optimistic default
    return True

# Spouse proving
#--------------------------------------------------------

def prove_spouses(c):
    # In practice it is fairly easy to prove spouse relationship.
    # We only run into problems where one partner could be classified
    # into multiple marriages due to optimistic spouse rule.
    # Here we deal with these ambiguous cases by preferring situtations
    # where the females name includes the male name. In pratice this
    # seems to easily give us definitive spouse relationships.

    # Find sets of individual joined by the possibility of marriage
    gb = graphlib.GraphBuilder()
    for e in c.edges:
        if not e.definitive_spouse and e.could_be_spouse:
            assert not e.a.spouse
            assert not e.b.spouse
            gb.add_edge(gb.get_node(e.a), gb.get_node(e.b))

    for cc in gb.get_graph().components:
        c.spouse_collisions = handle_spouse_collisions(cc)

def handle_spouse_collisions(c):
    assert len(c.nodes) >= 2

    people = [n.value for n in c.nodes]
    males = [n for n in people if n.a.sex == 0]
    females = [n for n in people if n.a.sex == 1]

    # Easy case, only possibility for these 2 people to be married
    if len(c.nodes) == 2:
        m, = males
        f, = females
        make_spouse(m, f)
        return

    while males and females:
        # Marry the couple that shares the largest substring in their main names.
        # e.g. Mrs. Sammuel Herman and Mr. Sammuel Herman
        n = np.array([0 if m.parsed_name.last != f.parsed_name.last else
                       largest_common_substring(m.parsed_name.main, f.parsed_name.main)
                      for m in males
                      for f in females])
        i = np.argmax(n)
        m_i,f_i = divmod(i, len(females))
        m = males.pop(m_i)
        f = females.pop(f_i)
        m.get_edge_to(f).make_definitive_spouse()

    return True

def make_spouse(m, f):
    assert not m.spouse
    assert not f.spouse
    e = m.get_edge_to(f)
    assert e.could_be_spouse
    assert not e.definitive_spouse
    e.set_all_possbilities(False)
    e.could_be_spouse = True
    m.spouse = f
    f.spouse = m


# Parent proving
#--------------------------------------------------------

def prove_parents(c):
    checked = set()
    for p in c.nodes:
        if p in checked:
            continue
        assert p.spouse not in checked
        parents = filter(None, [p, p.spouse])
        for p in parents:
            checked.add(p)
        prove_parents_children(parents)

def prove_parents_children(parents):
    assert not any(p.children for p in parents)
    n_max_children = max(p.adjusted_parch for p in parents)
    children = set()
    for p in parents:
        for e in p.edges:
            if not e.could_be_child:
                continue
            other = e.other(p)
            if not child_parent_direction(p, other):
                continue
            #if has_other_possible_parents(other, parents):
            #    continue
            children.add(other)

    if not children:
        return

    errors = False
    if len(children) > n_max_children:
        print 'warning: %d children when found when max was expected to be %d' % (
            len(children), n_max_children)
        errors = True

    if len(parents) == 1:
        mother = parents[0]
        father = None
    else:
        mother, father = parents
    if mother.sex == 0:
        mother,father = father,mother

    for c in children:
        if c.mother is not None:
            if c.mother != mother:
                print 'warning: inconsistent mother for child'
                errors = True
        if mother is not None and not c.has_edge_to(mother):
            print 'no relationship between mother and child'
        if c.father is not None:
            if c.father != father:
                print 'warning: inconsistent father for child'
                errors = True
        if father is not None and not c.has_edge_to(father):
            print 'no relationship between father and child'

    if errors:
        return

    for c in children:
        c.mother = mother
        c.father = father
        if mother:
            e = c.get_edge_to(mother)
            e.set_all_possbilities(False)
            e.could_be_child = True
            e.a = mother
            e.b = c
        if father:
            e = c.get_edge_to(father)
            e.set_all_possbilities(False)
            e.could_be_child = True
            e.a = father
            e.b = c

    ct = tuple(children)
    if mother:
        mother.children = ct
    if father:
        father.children = ct

#     for c in children:
#         other_children = children - set([c])
#         for o in other_children:
#             if id(c) < id(o):
#                 continue
#             e = c.get_edge_to(other)
#             assert e.could_be_sibling
#             e.set_all_possbilities(False)
#             e.could_be_sibling = True
#         c.siblings = tuple(other_children)

def child_parent_direction(parent, child):
    if parent.a.age != -1 and child.a.age != -1:
        return parent.age > child.age
    if parent.parsed_name.last != maiden_name(child):
        return False
    if parent.spouse:
        try:
            e = parent.get_edge_to(child)
        except ValueError:
            return False
        return e.could_be_child
    if child.parsed_name.title in ('miss','master'):
        return True
    if (parent.parsed_name.title in ('miss','master') and
        child.parsed_name.title in ('mrs','mr')):
        return False

    print 'difficult child parent direction'
    return parent.parch > child.parch

def has_other_possible_parents(child, parents):
    for e in child.edges:
        if not e.could_be_child
            continue
        if e.other(child) in parents:
            continue
        if child_parent_direction(e.other(child), child):
            return True
    return False

# Utilities
#--------------------------------------------------------

def share_name(a, b):
    return any(an==bn
               for an in a.parsed_name.iter_last_names()
               for bn in b.parsed_name.iter_last_names())

def general_affinity(a, b):
    return (ambiguous_equal(a.embarked, b.embarked) and
            ambiguous_equal(a.pclass, b.pclass))

def ambiguous_equal(a, b):
    return a == b or a < 0 or b < 0

def ambiguous_gt(a, b):
    return a<0 or b<0 or a>b

def ambiguous_lt(a, b):
    return a<0 or b<0 or a<b

def ambiguous_gt_diff(a, b, d):
    return a<0 or b<0 or a-b > d

def largest_common_substring(a, b):
    for i in xrange(min(len(a), len(b))):
        if a[:i:] != b[:i:]:
            break
    return i

def has_common_parents(a, b):
    return bool(set(a.get_known_parents()) & set(b.get_known_parents()))

def maiden_name(a):
    if a.sex == 1 and a.parsed_name.title == 'mrs' and a.parsed_name.other:
        return a.parsed_name.other.rsplit()[-1]
    return a.parsed_name.last



# Dot file creation
#--------------------------------------------------------

@contextmanager
def block(fp, name, *args):
    print >>fp, name, ' '.join(args), '{'
    yield
    print >>fp, '}'


class DotCreator(object):

    def __init__(self, fp):
        self.fp = fp
        self.graph_counter = 0

    def write_components(self, components, individual_digraphs=False):
        if not individual_digraphs:
            with block(fp, 'digraph', self.next_graph_name()):
                for c in components:
                    self.write_component(c)
        else:
            for c in components:
                with block(fp, 'digraph', self.next_graph_name()):
                    self.write_component(c)

    def next_graph_name(self):
        i = self.graph_counter
        self.graph_counter += 1
        return 'G%d' % (i,)

    def write_component(self, c):
        families, extra_nodes, extra_edges = find_nuclear_families(c)

def display_graph(path, components, show_extra=True):
    print 'writing', path
    write_dot('/tmp/family.dot', components, show_extra)
    run_program('./dotpack.sh', #'-v',
                '/tmp/family.dot', path
                )

def write_dot(path, cs, show_extra):
    # need to save references to famiies so that they don't disappear
    all_families = []
    with open(path, 'w') as fp:
#        print >>fp, 'rankdir=LR;'
        for i,c in enumerate(cs):
            print >>fp, 'digraph G%d {' % (i,)
            families, extra_nodes, extra_edges = find_families(c)
            all_families.append(families)
            for family in families:
                write_family(fp, family)
            for n in extra_nodes:
                write_node(fp, n)
            for e in extra_edges:
                if show_extra or e.wx < 0.01:
                    write_edge(fp, e, make_extra_edge(e))
            print >>fp, '}'

def write_family(fp, family):
    print >>fp, 'subgraph {'
    print >>fp, '%d [label="%s" shape="circle"]' % (id(family), family.name)

    for p,label in [(family.mother, 'mother'), (family.father, 'father')]:
        if not p:
            continue
        write_node(fp, p)
        print >>fp, '%d -> %d [label="%s"]' % (id(p), id(family), label)
    print >>fp, '{rank=same;',
    for p in family.mother, family.father:
        if p:
            print >>fp, id(p),
    print >>fp, '}'

    for c in family.children:
        if not c.write_elsewhere:
            write_node(fp, c)
        print >>fp, '%d -> %d [label="child"]' % (id(family), id(c))
    print >>fp, '{rank=same;',
    for c in family.children:
        print >>fp, id(c),
    print >>fp, '}'

    print >>fp, '}'

def write_edge(fp, e, attr):
    print >>fp, '%d -> %d [%s];' % (id(e.a), id(e.b), attr)

def write_node(fp, node):
    print >>fp, '%d [label="%s\\ns=%d p=%d c=%d e=%s c=%s\\na=%.1f f=%.1f" shape="rectangle" color=%s];' % (
        id(node), node.name.replace('"', '\\"'),
        node.sibsp, node.parch, node.pclass, node.embarked, node.cabin,
        node.age, node.fare,
        {True:'green', False:'red', None:'black'}[node.survived])

def make_extra_edge(e):
    weights = np.array([e.wp, e.wc, e.ws, e.wx])
    labels = np.array(['spouse', 'child', 'sibling', 'extended'])
    inx = np.argsort(weights)[::-1]
    weights = weights[inx]
    labels = labels[inx]
    W = np.cumsum(weights)

    penwidth = 1
    style = 'solid'
    if labels[0] == 'extended' and weights[0] > 0.01 + weights[1]:
        style = 'dashed'

    if weights[0] > 0.99:
        label=labels[0]
        if label != 'extended':
            penwidth = 1
        if label == 'child':
            if (e.a.father is e.b) or (e.b.father is e.a):
                label = 'father'
            elif (e.a.mother is e.b) or (e.b.mother is e.a):
                label = 'mother'
    else:
        mask = weights > 0.1
        label = ' '.join('%s=%.1f' % (l,w) for l,w in zip(labels[mask], weights[mask]))

    return 'label="%s" penwidth=%d style="%s"' % (label, penwidth, style)
