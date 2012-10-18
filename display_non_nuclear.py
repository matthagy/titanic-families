'''Scripts to display family trees using dot
'''

import os
import os.path
from subprocess import check_call

from graphlib import Component
from data import TitanicDataSet
from findfamilies import construct_family_components, DotCreator, find_nuclear_families


OUTPUT_DIR = 'nonnuclear_families_graphs'
DOT_TMP_PATH = '/tmp/families_graph_tmp.dot'
DOT_SCRIPT = './dotpack.sh'
MAX_FILE_NODES = 140

def main():
    train = TitanicDataSet.get_train()
    test = TitanicDataSet.get_test()
    families = construct_family_components(train, test)

    acc = []
    i = 0
    for c in families:
        #if len(c.nodes) == 1:
        #    continue
        nuclear_families, extra_nodes, extra_edges = find_nuclear_families(c)
        c.tear_down()
        acc.append(Component(extra_nodes, extra_edges))

        if sum(len(c.nodes) for c in acc) > 40:
            display_graph(i, acc)
            i += 1
            acc = []

    if acc:
        display_graph(i, acc)

def display_graph(i, components):
    print 'displaying', i
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    output_path = os.path.join(OUTPUT_DIR, '%d.png' % (i,))
    generate_graph(output_path, components)

def generate_graph(output_path, components):
    with open(DOT_TMP_PATH, 'w') as fp:
        dc = DotCreator(fp)
        #dc.show_extended = False
        dc.write_components(components,
                            individual_digraphs=True,
                            show_nuclear_families=False)
    ncols = determine_ncols(components)
    check_call([DOT_SCRIPT, DOT_TMP_PATH, str(ncols), output_path])

def determine_ncols(components):
    largest_component = max(len(c.nodes) for c in components)
    if largest_component > 8:
        return 3
    elif largest_component > 5:
        return 4
    elif largest_component > 3:
        return 6
    elif largest_component > 1:
        return 8
    return 10

__name__ == '__main__' and main()
