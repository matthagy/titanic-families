'''Show the simple structuer of close families (spouses and parent/child)
and how this impacted survivial.
'''

from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

from data import TitanicDataSet
from findfamilies import construct_family_components, find_nuclear_families

def main():
    plot_1st_class()
    plot_2nd_class()
    plot_3rd_class()

def plot_3rd_class():
    plot_class(1, '$3^\mathrm{rd}$', 2, 5.7)

def plot_2nd_class():
    plot_class(2, '$2^\mathrm{nd}$', 1, 3.7)

def plot_1st_class():
    plot_class(3, '$1^\mathrm{st}$', 0, 4.9)

def plot_class(fignum, name, pclass, y_max):
    def splice(ds):
        return ds.splice(ds.pclass == pclass)
    train = splice(TitanicDataSet.get_train())
    test = splice(TitanicDataSet.get_test())
    families = construct_family_components(train, test)
    families = sorted(families, key=lambda f: len(f.nodes))
    families = list(f for f in families if not f.difficult_parent_child)

    frames = []
    for f in families:
        nuclear_families, extra_nodes, extra_edges = find_nuclear_families(f)
        for nf in nuclear_families:
            frames.append(FamilyFrame(nf.mother, nf.father, nf.children))
        for e in extra_edges:
            if e.definitive_spouse:
                a,b = e.a, e.b
                if a.a.sex == 0:
                    a,b = b,a
                frames.append(CoupleFrame(a,b))
    frames.sort(key=lambda f: (not isinstance(f,CoupleFrame), f.n_members))

    for f in frames:
        f.setup()
        f.scale(1.1)

    fp = FramePlacer(11, 0.5, [0.5, 0.2])
    fp.place_frames(frames)

    plt.figure(fignum)
    plt.clf()
    for m,c,ps in fp.collect_points():
        plt.plot(ps[::, 0], ps[::, 1], linestyle='None', marker=m, color=c, ms=9)

    lines = LineCollection(fp.collect_lines(),
                           colors='k',
                           linestyles='solid')
    plt.gca().add_collection(lines)
    plt.title(name + 'Class Families')
    def label(label, **kwds):
        plt.plot([-1,-1], [-1,-1], label=label, **kwds)
    label('Female', marker='D', linestyle='None', markerfacecolor='white', color='k')
    label('Male', marker='o', linestyle='None', markerfacecolor='white', color='k')
    label('Survived', marker='s', linestyle='None', markerfacecolor=(0,1,0), markeredgecolor='white')
    label('Died', marker='s', linestyle='None', markerfacecolor='r', markeredgecolor='white')
    label('Unkown', marker='s', linestyle='None', markerfacecolor='k', markeredgecolor='white')
    plt.legend(loc='upper left', numpoints=1, frameon=False, ncol=3)
    plt.xlim(0, 12)
    plt.ylim(-0.1, y_max)
    plt.xticks([])
    plt.yticks([])
    plt.draw()
    plt.show()
    plt.savefig('%d_class_families.png' % (pclass+1,), bbox_inches='tight', pad_inches=0.1)


class MockPerson(object):
    '''Used in testing placement algorithm
    '''

    class Attributes(object):
        def __init__(self, sex):
            self.sex = sex

    def __init__(self, sex, survived):
        self.a = self.Attributes(sex)
        self.survived = survived


class BaseFrame(object):

    def setup(self):
        self.points = []
        self.lines = []
        self.create_frame()
        assert len(self.lines)
        assert len(self.points)
        self.lines = np.array(self.lines)
        self.markers, self.colors, points = zip(*self.points)
        self.points = np.array(points)

    def scale(self, factor):
        self.points *= factor
        self.lines *= factor

    def shift(self, offset):
        x,y = offset * np.ones((2,))
        self.points[::, 0] += x
        self.points[::, 1] += y
        self.lines[::, :2:] += x
        self.lines[::, 2::] += y

    def calculate_extent(self):
        x_min, y_min = self.points.min(axis=0)
        x_max, y_max = self.points.max(axis=0)
        return [x_min, x_max, y_min, y_max]

    def calculate_dimensions(self):
        x_min, x_max, y_min, y_max = self.calculate_extent()
        return [x_max - x_min, y_max - y_min]

    @property
    def width(self):
        return self.calculate_dimensions()[0]

    @property
    def height(self):
        return self.calculate_dimensions()[1]

    def create_line(self, xa, xb, ya, yb):
        self.lines.append([xa, xb, ya, yb])

    def create_vline(self, x, ya, yb):
        self.create_line(x, x, ya, yb)

    def create_hline(self, y, xa, xb):
        self.create_line(xa, xb, y, y)

    sex_marker_map = {
        0:'o',
        1:'D'}

    survived_color_map = {
        True:(0,1,0),
        False:'r',
        None:'k'}

    person_spacing = 0.5

    def create_person(self, x, y, p):
        self.points.append([self.sex_marker_map[p.a.sex],
                            self.survived_color_map[p.survived],
                            (x,y)])

    def create_2people(self, y, ppl):
        self.create_person(-0.5 * self.person_spacing, y, ppl[0])
        self.create_person(+0.5 * self.person_spacing, y, ppl[1])
        self.create_hline(y, -0.5 * self.person_spacing, +0.5* self.person_spacing)


class CoupleFrame(BaseFrame):

    def __init__(self, wife, husband):
        self.wife = wife
        self.husband = husband

    n_members = 2

    def create_frame(self):
        self.create_2people(0, [self.wife, self.husband])


class FamilyFrame(BaseFrame):

    def __init__(self, mother, father, children):
        assert mother or father
        assert children
        self.mother = mother
        self.father = father
        self.children = tuple(children)

    @property
    def n_members(self):
        return self.n_parents + self.n_children

    @property
    def n_parents(self):
        return sum(1 for p in (self.mother, self.father) if p)

    @property
    def n_children(self):
        return len(self.children)

    parent_child_offset = 0.25
    child_stem_length = 0.0

    def create_frame(self):
        self.create_vline(0, 0, -self.parent_child_offset)
        self.create_parents()
        self.create_children()

    def create_parents(self):
        parents = filter(None, [self.mother, self.father])
        assert parents
        if len(parents) == 1:
            self.create_person(0, 0, parents[0])
        else:
            self.create_2people(0, parents)

    def create_children(self):
        if self.n_children == 1:
            self.create_person(0, -self.parent_child_offset, self.children[0])
            return
        if self.n_children == 2:
            self.create_2people(-self.parent_child_offset, self.children)
            return

        x = self.person_spacing * (np.arange(self.n_children) - 0.5 * (self.n_children-1))
        self.create_hline(-self.parent_child_offset, x[0], x[-1])
        for xi,c in zip(x, self.children):
            y = -(self.parent_child_offset + self.child_stem_length)
            self.create_vline(xi, -self.parent_child_offset, y)
            self.create_person(xi, y, c)



class FramePlacer(object):

    def __init__(self, max_width, row_offset, padding):
        self.max_width = max_width
        self.row_offset = row_offset
        self.padding = padding
        self.x_offset = 0.0
        self.y_offset = 0.0
        self.frames = []
        self.rows = []
        self.row = []

    def place_frames(self, frames):
        frames = list(frames)
        while frames:
            for frame in frames:
                if self.can_place_frame_on_current_row(frame):
                    break
            else:
                frame = frames[0]
            frames.remove(frame)
            self.add_frame(frame)
        self.fixup_rows()

    def can_place_frame_on_current_row(self, frame):
        return frame.width + self.x_offset <= self.max_width

    def fixup_rows(self):
        for row in self.rows:
            self.fixup_row(row)
        if self.row:
            self.fixup_row(self.row)

    height_factor = 0.8

    def fixup_row(self, row):
        h = max(f.height for f in row)
        for f in row:
            f.shift([0, self.height_factor*(h - f.height)])

    def add_frame(self, frame):
        x_min,x_max, y_min, y_max = frame.calculate_extent()
        width = x_max - x_min
        if width + self.x_offset > self.max_width:
            self.rows.append(self.row)
            self.row = []
            self.y_offset += self.row_offset + self.padding[1]
            self.x_offset = 0.0
        frame.shift([-x_min + self.x_offset + self.padding[0],
                     -y_min + self.y_offset + self.padding[1]])

        self.x_offset += width + self.padding[0]
        self.frames.append(frame)
        self.row.append(frame)

    def collect_points(self):
        acc = defaultdict(list)
        for f in self.frames:
            for m,c,p in zip(f.markers, f.colors, f.points):
                acc[m,c].append(p)
        return [(m,c,np.array(ps)) for (m,c),ps in acc.iteritems()]

    def collect_lines(self):
        acc = []
        for f in self.frames:
            for xa,xb,ya,yb in f.lines:
                acc.append([(xa, ya), (xb, yb)])
        return np.array(acc)

__name__ == '__main__' and main()
