"""Heuristic-based algorithm to find the family trees in the titanic data.

Only involves a few parameters, all of which are related to age
(e.g. minimum age for marriage). See the constant defined at the
top for all of them. Neither iterative nor stochastic methods are
used.

First a graph of individuals is constructed where the edges represent
shared last names. This includes any previous names such maiden names.
Each edge represents a relationship that can be classified as one
of the following:
   o Spouse
   o Parent/Child
   o Sibling
   o Extended (e.g. aunt, cousin, or distant relative)

The classification scheme is optimistic, i.e. we only ask
whether or not the relationship is possible. Much of information
can be directly inferred from the given attributes (e.g. two
individual cannot be siblings if one of them has sibsp==0).

Next we prove spousal relationship. This fairly easy, epically
as many spouses name pairs are of the form:

   West, Mrs. Edwy Arthur (Ada Mary Worth)
   West, Mr. Edwy Arthur

We don't require names of this style and can also use age differences,
requires Mrs title for the female, and other simple heuristics.
The only difficulty arises when one individual could be classified as
married to multiple individuals. There are only a few such situations and
they can all be handled by assigning marriage to the couple in which the
female has the males first name (e.g. Mrs. Edwy Arthur).

With spousal relationships found it is then straightforward to workout
parent/child relationships. The only ambiguities at this point are
child vs. sibling and they can be resolved by checking for common
parent(s). Lastly, parent/child relationships can be used to work out
sibling relationships. We can then recover the structure of nuclear
families: families in which there is at least one parent and one or more
children.

Outside of the nuclear family structure, we still maintain the
relationship graph which allows for such classifications as:
  o siblings traveling together without any parents
  o extended relations
  o families joined by extended relationships

At the moment there are still some edge cases. In particular, the largest
relationship graph component isn't separated into a family structure.
Additionally, it would be nice to remove or relax the few parameters.
"""

from __future__ import division

import re
from contextlib import contextmanager
from collections import defaultdict

import numpy as np

from data import TitanicDataSet
import graphlib

__all__ = ['construct_family_components',
           'find_nuclear_families']


# Parameters
#--------------------------------------------------------
# No on under this age can be consider married
MINIMUM_AGE_FOR_MARRIAGE = 14

# # At most, a married woman can be n years older than husband
# # This rule isn't needed, and therefore have made the
# # value large enough to have no effect.
# MAXIMUM_MARRIED_FEMALE_AGE_ADVANTAGE = 100

# A parent must be at least n years older than their child
MINIMUM_PARENT_AGE_ADVANTAGE = 14

# # For two family graphs, there is an extra child-like individual
# # (possibly a niece or nephew or just distant relative) that can't
# # be discerned from the true children. With this set we classify them
# # as a child.
# ALLOW_ADDITIONAL_CHILDREN = True

# Name parsing
#--------------------------------------------------------

name_rgx = re.compile(r'''
^                 # Explicit start
       \s*
  ([^,]+)         # Last Name
     , \s+
  ([^.]+) \.      # Title
       \s+
  ([^("]+)?       # Main name
       \s*
  (?:
    "([^"]+)"     # Nick name
  )?
       \s*
  (?:             # Other name
     \(
        ([^)]+)
      \)
   )?
''', re.VERBOSE)


class ParsedName(object):

    def __init__(self, last, title, main, nick, other):
        if not main and other:
            main = other.split(None, 1)[0]
        self.last = last
        self.title = title
        self.main = main
        self.nick = nick
        self.other = other.strip('"') if other else None

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


# Graph data structures
#--------------------------------------------------------

class DotIDMixin(object):
    '''Base class for nodes that are written to dot files
    '''

    _dot_id_counter = 0

    @property
    def dot_id(self):
        try:
            return self._dot_id
        except AttributeError:
            DotIDMixin._dot_id_counter += 1
            self._dot_id = str(DotIDMixin._dot_id_counter)
        return self._dot_id


class Person(graphlib.Node, DotIDMixin):

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
        self.extendeds = ()

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
    def n_known_extended(self):
        return len(self.extendeds)

    def __str__(self):
        return 'p(%s)' % (self.name,)

    @property
    def adjusted_sibsp(self):
        return self.a.sibsp - len(self.siblings) - (1 if self.spouse else 0)

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
        self.set_all_possibilities(True)

    def set_all_possibilities(self, value):
        value = bool(value)
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
        return self.check_definitive(0)

    @property
    def definitive_sibling(self):
        return self.check_definitive(1)

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


# Relationship construction
#--------------------------------------------------------

class LastNameBuilder(graphlib.GraphBuilder):

    def __init__(self):
        super(LastNameBuilder, self).__init__(node_factory=CommonLastName,
                                              edge_factory=LastNameEdge)

    def add_edge(self, name, person):
        self.values_to_nodes[person] = person
        super(LastNameBuilder, self).add_edge(self.get_node(name),
                                              person)

def construct_family_components(train=TitanicDataSet.get_train(),
                                test=TitanicDataSet.get_test(),
                                tune=True):
    '''Entry point for finding relationships.

    Returns a list of graph components (graphlib.Component)
    where the nodes are individuals (Person) and edges are
    relationships (RelationEdge).
    '''
    lnb = LastNameBuilder()
    add_last_names(lnb, train)
    add_last_names(lnb, test)
    last_name_graph = lnb.get_graph()
    return [tune_family_relations(f) if tune else f
            for c in last_name_graph.components
            for f in build_relations(c)]

def add_last_names(nb, ds):
    if ds is None:
        return
    if ds.is_train:
        survived = ds.survived
    else:
        survived = [None] * len(ds)
    for survived,attributes in zip(survived, ds.iter_entries()):
        person = Person(attributes, survived)
        for last_name in person.parsed_name.iter_last_names():
            nb.add_edge(last_name, person)

def build_relations(c):
    # Extract people nodes, discarding LastNameNodes, and
    # clear all of the LastNameEdges
    nodes, edges = c.tear_down()
    people = [n for n in nodes if isinstance(n, Person)]
    for p in people:
        del p.edges[::]

    # Group together all of the people who share a last name
    # and meet general affinity qualifications
    gb = graphlib.GraphBuilder(edge_factory=RelationEdge)
    for p in people:
        gb.values_to_nodes[p] = p
    for i,a in enumerate(people):
        for b in people[i+1::]:
            if share_name(a, b) and general_affinity(a, b) != 0:
                gb.add_edge(a, b)
    return gb.get_graph().components

def find_nuclear_families(c):
    '''Finds the nuclear families in a graph component.

    Also returns any nodes and edges that are not included
    in families.
    '''
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
    if n.a.sex == 0:
        f.father = n
    else:
        f.mother = n
    seen.add(n)
    if n.spouse:
        if n.spouse.a.sex == 0:
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
    # Use simple heuristics to determine relationship individuals.
    # Throughout this procedure we incrementally prove various relationships
    # exists.
    update_relationship_possibilities(c)

    prove_spouses(c)
    update_relationship_possibilities(c)
    # ensure there is no ambiguity in spouse classification
    for e in c.edges:
        assert (not e.could_be_spouse) or e.definitive_spouse
    for n in c.nodes:
        n_spouse = sum(1 for e in n.edges if e.could_be_spouse)
        assert n_spouse in (0,1)

    prove_parents(c)
    if c.difficult_parent_child:
        return c
    update_relationship_possibilities(c)

    prove_siblings(c)
    update_relationship_possibilities(c)

    prove_extended(c)
    update_relationship_possibilities(c)

    return c

def update_relationship_possibilities(c):
    for e in c.edges:
        update_a_relationship_possibilities(e)

def update_a_relationship_possibilities(e):
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

def could_be_spouse(a, b):
    # already proven that they are spouses
    if a.spouse is not None and a.spouse is b:
        assert b.spouse is a
        return True
    # proven that they aren't spouses
    elif a.spouse or b.spouse:
        return False

    # check if sibsp or sex rules out possibility of them being spouse
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

    # check if the woman's main name includes a significant portion of
    # the mans name. (e.g Mrs. Sammuel Herman and Mr. Sammuel Herman)
    m,f = (a,b) if a.a.sex == 0 else (b,a)
    if m.parsed_name.main == f.parsed_name.main:
        return True
    n = largest_common_substring(m.parsed_name.main, f.parsed_name.main)
    if n > 0.5 * len(f.parsed_name.main):
        return True

    # rule out individual under MINIMUM_AGE_FOR_MARRIAGE
    if not (ambiguous_ge(a.a.age, MINIMUM_AGE_FOR_MARRIAGE) and
            ambiguous_ge(b.a.age, MINIMUM_AGE_FOR_MARRIAGE)):
        return False

#     # If we know both ages, rule out women who are 5+ years older than men
#     # this might be an invalid rule as some of entries look like older rich
#     # women married to young men. The cases I've observed are already handled
#     # by the earlier check if the woman's main name includes the main.
#     # This rule was added to help sort out the mother/son vs. mother/husband,
#     # but it may be better to leave those cases ambiguous anyways.
#     if not ambiguous_le_diff(f.a.age, m.a.age, MAXIMUM_MARRIED_FEMALE_AGE_ADVANTAGE):
#         return False

    # optimistic default
    return True

def could_be_child(parent, child):
    # check if we've already proven that parent or their spouse is
    # a parent to this child
    if set(child.known_parents) & set(filter(None, [parent, parent.spouse])):
        return True

    if parent.spouse is not None and parent.spouse is child:
        assert child.spouse is parent
        return False

    # check if parch rules out possibility of parent/child relationship
    if parent.adjusted_parch == 0 or child.adjusted_parch == 0:
        return False
    # rule out possibility of parents conceiving a child when under
    # a certain age
    if not ambiguous_ge_diff(parent.a.age, child.a.age, MINIMUM_PARENT_AGE_ADVANTAGE):
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

def could_be_sibling(a, b):
    # Already proven that they are siblings
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
    # Here we deal with these ambiguous cases by preferring situations
    # where the females name includes the male name. In practice this
    # seems to easily give us definitive spouse relationships.

    # Find sets of individual joined by the possibility of marriage
    gb = graphlib.GraphBuilder()
    for e in c.edges:
        if e.could_be_spouse:
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
    if len(people) == 2:
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
        make_spouse(m, f)

    return True

def make_spouse(m, f):
    assert not m.spouse
    assert not f.spouse
    e = m.get_edge_to(f)
    assert e.could_be_spouse
    e.set_all_possibilities(False)
    e.could_be_spouse = True
    m.spouse = f
    f.spouse = m


# Parent proving
#--------------------------------------------------------

def prove_parents(c):
    checked = set()
    c.difficult_parent_child = False
    for p in c.nodes:
        if p in checked:
            continue
        assert p.spouse not in checked
        parents = filter(None, [p, p.spouse])
        for p in parents:
            checked.add(p)
        prove_parents_children(c, parents)

def prove_parents_children(comp, parents):
    assert not any(p.children for p in parents)
    n_max_children = min(p.adjusted_parch for p in parents)
    if n_max_children == 0:
        return

    children = set()
    for p in parents:
        for e in p.edges:
            if not e.definitive_child:
                continue
            other = e.other(p)
            if not child_parent_direction(p, other):
                continue
            #if has_other_possible_parents(other, parents):
            #    continue
            children.add(other)

    if not children:
        return

    if len(children) > n_max_children:
        children = discern_children_by_fare(parents, children, n_max_children)

    errors = False
    if len(children) > n_max_children:
        print 'warning: %d children were found when max was expected to be %d' % (
            len(children), n_max_children)
#         for p in parents:
#             print p.a.name
#         print '-'*60
#         for c in children:
#             print c.parsed_name.main
#         print
        errors = True

    if len(parents) == 1:
        mother = parents[0]
        father = None
    else:
        mother, father = parents
    if mother.a.sex == 0:
        mother,father = father,mother

    for c in children:
        if c.mother is not None:
            if c.mother != mother:
                print 'warning: inconsistent mother for child'
                errors = True
        if mother is not None and not c.has_edge_to(mother):
            print 'no relationship between mother and child'
            errors = True
        if c.father is not None:
            if c.father != father:
                print 'warning: inconsistent father for child'
                errors = True
        if father is not None and not c.has_edge_to(father):
            print 'no relationship between father and child'
            errors = True

    if errors:
        comp.difficult_parent_child = True
        return

    for c in children:
        c.mother = mother
        c.father = father
        if mother:
            e = c.get_edge_to(mother)
            e.set_all_possibilities(False)
            e.could_be_child = True
            e.a = mother
            e.b = c
        if father:
            e = c.get_edge_to(father)
            e.set_all_possibilities(False)
            e.could_be_child = True
            e.a = father
            e.b = c

    ct = frozenset(children)
    if mother:
        mother.children = ct
    if father:
        father.children = ct

# In practice this function only helps with one specific case
def discern_children_by_fare(parents, children, n_max_children):
    return [c for c in children
            if any(np.allclose(c.a.fare, p.a.fare)
                   for p in parents)]

def child_parent_direction(parent, child):
    if parent.a.age != -1 and child.a.age != -1:
        return parent.a.age > child.a.age
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
    return parent.a.parch > child.a.parch

def has_other_possible_parents(child, parents):
    for e in child.edges:
        if not e.could_be_child:
            continue
        if e.other(child) in parents:
            continue
        if child_parent_direction(e.other(child), child):
            return True
    return False


# Sibling proving
#--------------------------------------------------------
def prove_siblings(c):
    # at this stage, most sibling relationships have been worked out
    # as we've already worked spouses and parent/child relationships

    prove_symmetric(c, 'definitive_sibling', 'siblings')

def prove_symmetric(c, e_attr, col_attr):

    acc = defaultdict(list)
    for p in c.nodes:
        for e in p.edges:
            if getattr(e, e_attr):
                acc[p].append(e.other(p))

    for p in c.nodes:
        setattr(p, col_attr, frozenset(acc[p]))

    for p in c.nodes:
        for o in getattr(p, col_attr):
            assert p in getattr(o, col_attr)

# Sibling proving
#--------------------------------------------------------
def prove_extended(c):
    prove_symmetric(c, 'definitive_extended', 'extendeds')


# Utilities
#--------------------------------------------------------

def share_name(a, b):
    return any(an==bn
               for an in a.parsed_name.iter_last_names()
               for bn in b.parsed_name.iter_last_names())

def general_affinity(a, b):
    return (ambiguous_equal(a.a.embarked, b.a.embarked) and
            ambiguous_equal(a.a.pclass, b.a.pclass))

def ambiguous_equal(a, b):
    return a == b or a < 0 or b < 0

def ambiguous_gt(a, b):
    return a<0 or b<0 or a>b

def ambiguous_ge(a, b):
    return a<0 or b<0 or a>=b

def ambiguous_lt(a, b):
    return a<0 or b<0 or a<b

def ambiguous_gt_diff(a, b, d):
    return a<0 or b<0 or a-b > d

def ambiguous_ge_diff(a, b, d):
    return a<0 or b<0 or a-b >= d

def ambiguous_le_diff(a, b, d):
    return a<0 or b<0 or a-b <= d

def largest_common_substring(a, b):
    for i in xrange(min(len(a), len(b))):
        if a[:i:] != b[:i:]:
            break
    return i

def has_common_parents(a, b):
    return bool(set(a.known_parents) & set(b.known_parents))

def maiden_name(p):
    if p.a.sex == 1 and p.parsed_name.title == 'mrs' and p.parsed_name.other:
        return p.parsed_name.other.rsplit(None, 1)[-1]
    return p.parsed_name.last



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
        self.written_nodes = set()

    def write_components(self, components, individual_digraphs=False, **kwds):
        if not individual_digraphs:
            with block(self.fp, 'digraph', self.next_graph_name()):
                for c in components:
                    self.write_component(c, **kwds)
        else:
            for c in components:
                with block(self.fp, 'digraph', self.next_graph_name()):
                    self.write_component(c, **kwds)

    def next_graph_name(self):
        i = self.graph_counter
        self.graph_counter += 1
        return 'G%d' % (i,)

    def write_component(self, c, show_nuclear_families=True):
        if show_nuclear_families:
            self.write_nuclear_families(c)
            return

        for n in c.nodes:
            self.write_common_node(n)
        for e in c.edges:
            self.write_common_edge(e)

    def write_nuclear_families(self, c):
        families, extra_nodes, extra_edges = find_nuclear_families(c)
        for f in families:
            self.write_family(f)
        for n in extra_nodes:
            self.write_common_node(n)
        for e in extra_edges:
            self.write_common_edge(e)

    def write_family(self, f):
        print >>self.fp, '%s [label="%s" shape="circle"]' % (f.dot_id, f.name)
        for p,label in [(f.mother, 'mother'), (f.father, 'father')]:
            if p:
                self.write_common_node(p)
                print >>self.fp, '%s -> %s [label="%s"]' % (p.dot_id, f.dot_id, label)
        print >>self.fp, '{rank=same;',
        for p in f.mother, f.father:
            if p:
                print >>self.fp, p.dot_id,
        print >>self.fp, '}'

        for c in f.children:
            self.write_common_node(c)
            print >>self.fp, '%s -> %s [label="child"]' % (f.dot_id, c.dot_id)

        print >>self.fp, '{rank=same;',
        for c in f.children:
            print >>self.fp, c.dot_id,
        print >>self.fp, '}'

    def write_common_node(self, n):
        if n in self.written_nodes:
            return
        self.write_node(n,
                   label=self.get_node_label(n),
                   color=self.get_node_color(n),
                   shape=self.get_node_shape(n),
                   **self.get_extra_node_attributes(n))

    def get_node_label(self, n):
        return '%s\\ns=%d p=%d c=%d e=%s c=%s\\na=%.1f f=%.1f' % (
            n.a.name, n.a.sibsp, n.a.parch, n.a.pclass, n.a.embarked, n.a.cabin,
            n.a.age, n.a.fare)

    def get_node_color(self, n):
        return {True:'green', False:'red', None:'black'}[n.survived]

    def get_node_shape(self, n):
        return 'rectangle'

    def get_extra_node_attributes(self, n):
        return {}

    show_extended = True
    def write_common_edge(self, e):
        if not e.definitive_extended or self.show_extended:
            self.write_edge(e, label='/'.join(l for l,v in zip(['spouse','sibling','child','extended'],
                                                               e.possibilities)
                                              if v),
                            style='solid' if not e.definitive_extended else 'dashed')

    def write_node(self, n, **kwds):
        assert n not in self.written_nodes
        print >>self.fp, '%s [%s]' % (n.dot_id, self.make_attributes(kwds))
        self.written_nodes.add(n)

    def write_edge(self, e, **kwds):
        print >>self.fp, '%s -> %s [%s]' % (e.a.dot_id, e.b.dot_id, self.make_attributes(kwds))

    @classmethod
    def make_attributes(cls, kwds):
        return ' '.join('%s=%s' % (k, cls.quote_value(k,v))
                        for k,v in kwds.iteritems())

    @staticmethod
    def quote_value(k, v):
        if k in ('label','shape'):
            return '"%s"' % (str(v).replace('"', '\\"'))
        return str(v)



