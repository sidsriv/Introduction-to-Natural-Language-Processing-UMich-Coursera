import sys

from providedcode.transitionparser import TransitionParser
from providedcode.dependencygraph import DependencyGraph


def handle_input(input_file, model_file):
    tp = TransitionParser.load(model_file)
    for line in input_file:
        sentence = DependencyGraph.from_sentence(line)
        parsed = tp.parse([sentence])
        print parsed[0].to_conll(10).encode('utf-8')


if __name__ == '__main__':
    assert len(sys.argv) == 2, 'Usage: parse.py path_to.model'
    handle_input(sys.stdin, sys.argv[1])
