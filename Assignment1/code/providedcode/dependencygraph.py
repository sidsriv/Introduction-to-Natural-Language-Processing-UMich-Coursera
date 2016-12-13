# Natural Language Toolkit: Dependency Grammars
#
# Copyright (C) 2001-2015 NLTK Project
# Author: Jason Narad <jason.narad@gmail.com>
#         Steven Bird <stevenbird1@gmail.com> (modifications)
#         Robert Ying <robert.ying@columbia.edu> (modifications)
#
# URL: <http://nltk.org/>
# For license information, see LICENSE.TXT
#

"""
Tools for reading and writing dependency trees.
The input is assumed to be in Malt-TAB format
(http://stp.lingfil.uu.se/~nivre/research/MaltXML.html).
"""
from __future__ import print_function, unicode_literals

from collections import defaultdict
from itertools import chain
from pprint import pformat

from nltk.tree import Tree
from nltk.compat import python_2_unicode_compatible, string_types
import nltk

#################################################################
# DependencyGraph Class
#################################################################


@python_2_unicode_compatible
class DependencyGraph(object):

    @staticmethod
    def from_sentence(sent):
        tokens = nltk.word_tokenize(sent)
        tagged = nltk.pos_tag(tokens)
        
        dg = DependencyGraph()
        for (index, (word, tag)) in enumerate(tagged):
            dg.nodes[index + 1] = {
                'word': word,
                'lemma': '_',
                'ctag': tag,
                'tag': tag,
                'feats': '_',
                'rel': '_',
                'deps': defaultdict(),
                'head': '_',
                'address': index + 1,
            }
        dg.connect_graph()
        
        return dg
        
    """
    A container for the nodes and labelled edges of a dependency structure.
    """

    def __init__(self, tree_str=None, cell_extractor=None, zero_based=False, cell_separator=None):
        """Dependency graph.

        We place a dummy `TOP` node with the index 0, since the root node is
        often assigned 0 as its head. This also means that the indexing of the
        nodes corresponds directly to the Malt-TAB format, which starts at 1.

        If zero-based is True, then Malt-TAB-like input with node numbers
        starting at 0 and the root node assigned -1 (as produced by, e.g.,
        zpar).

        :param str cell_separator: the cell separator. If not provided, cells
        are split by whitespace.

        """
        self.nodes = defaultdict(lambda: {'deps': defaultdict(list)})
        self.nodes[0].update(
            {
                'word': None,
                'lemma': None,
                'ctag': 'TOP',
                'tag': 'TOP',
                'feats': None,
                'rel': 'TOP',
                'address': 0,
            }
        )

        self.root = None

        if tree_str:
            self._parse(
                tree_str,
                cell_extractor=cell_extractor,
                zero_based=zero_based,
                cell_separator=cell_separator,
            )

    def remove_by_address(self, address):
        """
        Removes the node with the given address.  References
        to this node in others will still exist.
        """
        del self.nodes[address]

    def redirect_arcs(self, originals, redirect):
        """
        Redirects arcs to any of the nodes in the originals list
        to the redirect node address.
        """
        for node in self.nodes.values():
            new_deps = []
            for dep in node['deps']:
                if dep in originals:
                    new_deps.append(redirect)
                else:
                    new_deps.append(dep)
            node['deps'] = new_deps

    def add_arc(self, head_address, mod_address):
        """
        Adds an arc from the node specified by head_address to the
        node specified by the mod address.
        """
        relation = self.nodes[mod_address]['rel']
        self.nodes[head_address]['deps'].setdefault(relation,[])
        self.nodes[head_address]['deps'][relation].append(mod_address)
        #self.nodes[head_address]['deps'].append(mod_address)


    def connect_graph(self):
        """
        Fully connects all non-root nodes.  All nodes are set to be dependents
        of the root node.
        """
        for node1 in self.nodes.values():
            for node2 in self.nodes.values():
                if node1['address'] != node2['address'] and node2['rel'] != 'TOP':
                    relation = node2['rel']
                    node1['deps'].setdefault(relation,[])
                    node1['deps'][relation].append(node2['address'])
                    #node1['deps'].append(node2['address'])

    def get_by_address(self, node_address):
        """Return the node with the given address."""
        return self.nodes[node_address]

    def contains_address(self, node_address):
        """
        Returns true if the graph contains a node with the given node
        address, false otherwise.
        """
        return node_address in self.nodes

    def __str__(self):
        return pformat(self.nodes)

    def __repr__(self):
        return "<DependencyGraph with {0} nodes>".format(len(self.nodes))

    @staticmethod
    def load(filename, zero_based=False, cell_separator=None):
        """
        :param filename: a name of a file in Malt-TAB format
        :param zero_based: nodes in the input file are numbered starting from 0
        rather than 1 (as produced by, e.g., zpar)
        :param str cell_separator: the cell separator. If not provided, cells
        are split by whitespace.

        :return: a list of DependencyGraphs

        """
        with open(filename) as infile:
            return [
                DependencyGraph(
                    tree_str,
                    zero_based=zero_based,
                    cell_separator=cell_separator,
                )
                for tree_str in infile.read().split('\n\n')
            ]

    def left_children(self, node_index):
        """
        Returns the number of left children under the node specified
        by the given address.
        """
        children = self.nodes[node_index]['deps']
        index = self.nodes[node_index]['address']
        return sum(1 for c in children if c < index)

    def right_children(self, node_index):
        """
        Returns the number of right children under the node specified
        by the given address.
        """
        children = self.nodes[node_index]['deps']
        index = self.nodes[node_index]['address']
        return sum(1 for c in children if c > index)

    def add_node(self, node):
        if not self.contains_address(node['address']):
            self.nodes[node['address']].update(node)

    def _parse(self, input_, cell_extractor=None, zero_based=False, cell_separator=None):
        """Parse a sentence.

        :param extractor: a function that given a tuple of cells returns a
        7-tuple, where the values are ``word, lemma, ctag, tag, feats, head,
        rel``.

        :param str cell_separator: the cell separator. If not provided, cells
        are split by whitespace.

        """

        def extract_3_cells(cells):
            word, tag, head = cells
            return word, word, tag, tag, '', head, ''

        def extract_4_cells(cells):
            word, tag, head, rel = cells
            return word, word, tag, tag, '', head, rel

        def extract_10_cells(cells):
            _, word, lemma, ctag, tag, feats, head, rel, _, _ = cells
            return word, lemma, ctag, tag, feats, head, rel

        extractors = {
            3: extract_3_cells,
            4: extract_4_cells,
            10: extract_10_cells,
        }

        if isinstance(input_, string_types):
            input_ = (line for line in input_.split('\n'))

        lines = (l.rstrip() for l in input_)
        lines = (l for l in lines if l)

        for index, line in enumerate(lines, start=1):
            cells = line.split(cell_separator)
            nrCells = len(cells)

            if cell_extractor is None:
                try:
                    cell_extractor = extractors[nrCells]
                except KeyError:
                    raise ValueError(
                        'Number of tab-delimited fields ({0}) not supported by '
                        'CoNLL(10) or Malt-Tab(4) format'.format(nrCells)
                    )

            word, lemma, ctag, tag, feats, head, rel = cell_extractor(cells)

            head = int(head)
            if zero_based:
                head += 1

            self.nodes[index].update(
                {
                    'address': index,
                    'word': word,
                    'lemma': lemma,
                    'ctag': ctag,
                    'tag': tag,
                    'feats': feats,
                    'head': head,
                    'rel': rel,
                }
            )

            self.nodes[head]['deps'][rel].append(index)

        if not self.nodes[0]['deps']['ROOT']:
            raise DependencyGraphError(
                "The graph does'n contain a node "
                "that depends on the root element."
            )
        root_address = self.nodes[0]['deps']['ROOT'][0]
        self.root = self.nodes[root_address]

    def _word(self, node, filter=True):
        w = node['word']
        if filter:
            if w != ',':
                return w
        return w

    def _tree(self, i):
        """ Turn dependency graphs into NLTK trees.

        :param int i: index of a node
        :return: either a word (if the indexed node is a leaf) or a ``Tree``.
        """
        node = self.get_by_address(i)
        word = node['word']
        deps = list(chain.from_iterable(node['deps'].values()))

        if deps:
            return Tree(word, [self._tree(dep) for dep in deps])
        else:
            return word

    def tree(self):
        """
        Starting with the ``root`` node, build a dependency tree using the NLTK
        ``Tree`` constructor. Dependency labels are omitted.
        """
        node = self.root

        word = node['word']
        deps = chain.from_iterable(node['deps'].values())
        return Tree(word, [self._tree(dep) for dep in deps])

    def triples(self, node=None):
        """
        Extract dependency triples of the form:
        ((head word, head tag), rel, (dep word, dep tag))
        """

        if not node:
            node = self.root

        head = (node['word'], node['ctag'])
        for i in node['deps']:
            dep = self.get_by_address(i)
            yield (head, dep['rel'], (dep['word'], dep['ctag']))
            for triple in self.triples(node=dep):
                yield triple

    def _hd(self, i):
        try:
            return self.nodes[i]['head']
        except IndexError:
            return None

    def _rel(self, i):
        try:
            return self.nodes[i]['rel']
        except IndexError:
            return None

    # what's the return type?  Boolean or list?
    def contains_cycle(self):
        """Check whether there are cycles.

        >>> dg = DependencyGraph(treebank_data)
        >>> dg.contains_cycle()
        False

        >>> cyclic_dg = DependencyGraph()
        >>> top = {'word': None, 'deps': [1], 'rel': 'TOP', 'address': 0}
        >>> child1 = {'word': None, 'deps': [2], 'rel': 'NTOP', 'address': 1}
        >>> child2 = {'word': None, 'deps': [4], 'rel': 'NTOP', 'address': 2}
        >>> child3 = {'word': None, 'deps': [1], 'rel': 'NTOP', 'address': 3}
        >>> child4 = {'word': None, 'deps': [3], 'rel': 'NTOP', 'address': 4}
        >>> cyclic_dg.nodes = {
        ...     0: top,
        ...     1: child1,
        ...     2: child2,
        ...     3: child3,
        ...     4: child4,
        ... }
        >>> cyclic_dg.root = top

        >>> cyclic_dg.contains_cycle()
        [3, 1, 2, 4]

        """
        distances = {}

        for node in self.nodes.values():
            for dep in node['deps']:
                key = tuple([node['address'], dep])
                distances[key] = 1

        for _ in self.nodes:
            new_entries = {}

            for pair1 in distances:
                for pair2 in distances:
                    if pair1[1] == pair2[0]:
                        key = tuple([pair1[0], pair2[1]])
                        new_entries[key] = distances[pair1] + distances[pair2]

            for pair in new_entries:
                distances[pair] = new_entries[pair]
                if pair[0] == pair[1]:
                    path = self.get_cycle_path(self.get_by_address(pair[0]), pair[0])
                    return path

        return False  # return []?

    def get_cycle_path(self, curr_node, goal_node_index):
        for dep in curr_node['deps']:
            if dep == goal_node_index:
                return [curr_node['address']]
        for dep in curr_node['deps']:
            path = self.get_cycle_path(self.get_by_address(dep), goal_node_index)
            if len(path) > 0:
                path.insert(0, curr_node['address'])
                return path
        return []

    def to_conll(self, style):
        """
        The dependency graph in CoNLL format.

        :param style: the style to use for the format (3, 4, 10 columns)
        :type style: int
        :rtype: str
        """

        if style == 3:
            template = '{word}\t{tag}\t{head}\n'
        elif style == 4:
            template = '{word}\t{tag}\t{head}\t{rel}\n'
        elif style == 10:
            template = '{i}\t{word}\t{lemma}\t{ctag}\t{tag}\t{feats}\t{head}\t{rel}\t_\t_\n'
        else:
            raise ValueError(
                'Number of tab-delimited fields ({0}) not supported by '
                'CoNLL(10) or Malt-Tab(4) format'.format(style)
            )

        return ''.join(template.format(i=i, **node) for i, node in sorted(self.nodes.items()) if node['tag'] != 'TOP')

    def nx_graph(self):
        """Convert the data in a ``nodelist`` into a networkx labeled directed graph."""
        import networkx as NX

        nx_nodelist = list(range(1, len(self.nodes)))
        nx_edgelist = [
            (n, self._hd(n))
            for n in nx_nodelist if self._hd(n)
        ]
        nx_labels = {}
        for n in nx_nodelist:
            nx_labels[n] = self.nodes[n]['word']

        g = NX.DiGraph()
        g.add_nodes_from(nx_nodelist)
        g.add_edges_from(nx_edgelist)

        return g, nx_labels


class DependencyGraphError(Exception):
    """Dependency graph exception."""
