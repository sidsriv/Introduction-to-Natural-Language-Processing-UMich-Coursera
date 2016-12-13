# Code modified from the Natural Language Toolkit
# Original author: Long Duong <longdt219@gmail.com>

import tempfile
import pickle
import os
import copy
import operator
import scipy.sparse as sparse
import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn import svm

class Configuration(object):
    """
    Class for holding configuration which is the partial analysis of the input sentence.
    The transition based parser aims at finding set of operators that transfer the initial
    configuration to the terminal configuration.

    The configuration includes:
        - Stack: for storing partially proceeded words
        - Buffer: for storing remaining input words
        - Set of arcs: for storing partially built dependency tree

    This class also provides a method to represent a configuration as list of features.
    """

    def __init__(self, dep_graph, feature_extractor):
        """
        :param dep_graph: the representation of an input in the form of dependency graph.
        :type dep_graph: DependencyGraph where the dependencies are not specified.
        :param feature_extractor: a function which operates on tokens, the
            stack, the buffer and returns a list of string features
        """
        # dep_graph.nodes contain list of token for a sentence
        self.stack = [0]  # The root element
        self.buffer = range(1, len(dep_graph.nodes))  # The rest is in the buffer
        self.arcs = []  # empty set of arc
        self._tokens = dep_graph.nodes
        self._max_address = len(self.buffer)

        self._user_feature_extractor = feature_extractor

    def __str__(self):
        return 'Stack : ' + \
            str(self.stack) + '  Buffer : ' + str(self.buffer) + '   Arcs : ' + str(self.arcs)

    def extract_features(self):
        """
        Extracts features from the configuration
        :return: list(str)
        """
        return self._user_feature_extractor(self._tokens, self.buffer, self.stack, self.arcs)

class TransitionParser(object):
    """
    An arc-eager transition parser
    """

    def __init__(self, transition, feature_extractor):
        self._dictionary = {}
        self._transition = {}
        self._match_transition = {}
        self._model = None
        self._user_feature_extractor = feature_extractor
        self.transitions = transition

    def _get_dep_relation(self, idx_parent, idx_child, depgraph):
        p_node = depgraph.nodes[idx_parent]
        c_node = depgraph.nodes[idx_child]

        if c_node['word'] is None:
            return None  # Root word

        if c_node['head'] == p_node['address']:
            return c_node['rel']
        else:
            return None

    def _convert_to_binary_features(self, features):
        """
        This function converts a feature into libsvm format, and adds it to the
        feature dictionary
        :param features: list of feature string which is needed to convert to
            binary features
        :type features: list(str)
        :return : string of binary features in libsvm format  which is
            'featureID:value' pairs
        """
        unsorted_result = []
        for feature in features:
            self._dictionary.setdefault(feature, len(self._dictionary))
            unsorted_result.append(self._dictionary[feature])

        # Default value of each feature is 1.0
        return ' '.join(str(featureID) + ':1.0' for featureID in sorted(unsorted_result))

    @staticmethod
    def _is_projective(depgraph):
        """
        Checks if a dependency graph is projective
        """
        arc_list = set()
        for key in depgraph.nodes:
            node = depgraph.nodes[key]
            if 'head' in node:
                childIdx = node['address']
                parentIdx = node['head']
                arc_list.add((parentIdx, childIdx))

        for (parentIdx, childIdx) in arc_list:
            # Ensure that childIdx < parentIdx
            if childIdx > parentIdx:
                temp = childIdx
                childIdx = parentIdx
                parentIdx = temp
            for k in range(childIdx + 1, parentIdx):
                for m in range(len(depgraph.nodes)):
                    if (m < childIdx) or (m > parentIdx):
                        if (k, m) in arc_list:
                            return False
                        if (m, k) in arc_list:
                            return False
        return True

    def _write_to_file(self, key, binary_features, input_file):
        """
        write the binary features to input file and update the transition dictionary
        """
        self._transition.setdefault(key, len(self._transition) + 1)
        self._match_transition[self._transition[key]] = key

        input_str = str(self._transition[key]) + ' ' + binary_features + '\n'
        input_file.write(input_str.encode('utf-8'))

    def _create_training_examples_arc_eager(self, depgraphs, input_file):
        """
        Create the training example in the libsvm format and write it to the input_file.
        Reference : 'A Dynamic Oracle for Arc-Eager Dependency Parsing' by Joav Goldberg and Joakim Nivre
        """
        training_seq = []

        projective_dependency_graphs = [dg for dg in depgraphs if TransitionParser._is_projective(dg)]
        countProj = len(projective_dependency_graphs)

        for depgraph in projective_dependency_graphs:
            conf = Configuration(depgraph, self._user_feature_extractor.extract_features)

            while conf.buffer:
                b0 = conf.buffer[0]
                features = conf.extract_features()
                binary_features = self._convert_to_binary_features(features)

                if conf.stack:
                    s0 = conf.stack[-1]
                    # Left-arc operation
                    rel = self._get_dep_relation(b0, s0, depgraph)
                    if rel is not None:
                        key = self.transitions.LEFT_ARC + ':' + rel
                        self._write_to_file(key, binary_features, input_file)
                        self.transitions.left_arc(conf, rel)
                        training_seq.append(key)
                        continue

                    # Right-arc operation
                    rel = self._get_dep_relation(s0, b0, depgraph)
                    if rel is not None:
                        key = self.transitions.RIGHT_ARC + ':' + rel
                        self._write_to_file(key, binary_features, input_file)
                        self.transitions.right_arc(conf, rel)
                        training_seq.append(key)
                        continue

                    # reduce operation
                    flag = False
                    for k in range(s0):
                        if self._get_dep_relation(k, b0, depgraph) is not None:
                            flag = True
                        if self._get_dep_relation(b0, k, depgraph) is not None:
                            flag = True

                    if flag:
                        key = self.transitions.REDUCE
                        self._write_to_file(key, binary_features, input_file)
                        self.transitions.reduce(conf)
                        training_seq.append(key)
                        continue

                # Shift operation as the default
                key = self.transitions.SHIFT
                self._write_to_file(key, binary_features, input_file)
                self.transitions.shift(conf)
                training_seq.append(key)

        print(" Number of training examples : {}".format(len(depgraphs)))
        print(" Number of valid (projective) examples : {}".format(countProj))
        return training_seq

    def train(self, depgraphs):
        """
        :param depgraphs : list of DependencyGraph as the training data
        :type depgraphs : DependencyGraph
        """

        try:
            input_file = tempfile.NamedTemporaryFile(
                prefix='transition_parse.train',
                dir=tempfile.gettempdir(),
                delete=False)

            self._create_training_examples_arc_eager(depgraphs, input_file)

            input_file.close()
            # Using the temporary file to train the libsvm classifier
            x_train, y_train = load_svmlight_file(input_file.name)
            # The parameter is set according to the paper:
            # Algorithms for Deterministic Incremental Dependency Parsing by Joakim Nivre
            # this is very slow.
            self._model = svm.SVC(
                kernel='poly',
                degree=2,
                coef0=0,
                gamma=0.2,
                C=0.5,
                verbose=False,
                probability=True)

            print('Training support vector machine...')
            self._model.fit(x_train, y_train)
            print('done!')
        finally:
            os.remove(input_file.name)

    def parse(self, depgraphs):
        """
        :param depgraphs: the list of test sentence, each sentence is represented as a dependency graph where the 'head' information is dummy
        :type depgraphs: list(DependencyGraph)
        :return: list (DependencyGraph) with the 'head' and 'rel' information
        """
        result = []
        if not self._model:
            raise ValueError('No model trained!')

        for depgraph in depgraphs:
            conf = Configuration(depgraph, self._user_feature_extractor.extract_features)
            while conf.buffer:
                features = conf.extract_features()
                col = []
                row = []
                data = []
                for feature in features:
                    if feature in self._dictionary:
                        col.append(self._dictionary[feature])
                        row.append(0)
                        data.append(1.0)
                np_col = np.array(sorted(col))  # NB : index must be sorted
                np_row = np.array(row)
                np_data = np.array(data)

                x_test = sparse.csr_matrix((np_data, (np_row, np_col)), shape=(1, len(self._dictionary)))

                pred_prob = self._model.predict_proba(x_test)[0]

                sorted_predicted_values = [
                    self._model.classes_[x[0]]
                    for x in sorted(enumerate(pred_prob), key=operator.itemgetter(1), reverse=True)]

                # Note that SHIFT is always a valid operation
                for y_pred in sorted_predicted_values:
                    if y_pred in self._match_transition:
                        strTransition = self._match_transition[y_pred]
                        try:
                            baseTransition, relation = strTransition.split(":")
                        except ValueError:
                            baseTransition = strTransition

                        if baseTransition == self.transitions.LEFT_ARC:
                            if self.transitions.left_arc(conf, relation) != -1:
                                break
                        elif baseTransition == self.transitions.RIGHT_ARC:
                            if self.transitions.right_arc(conf, relation) != -1:
                                break
                        elif baseTransition == self.transitions.REDUCE:
                            if self.transitions.reduce(conf) != -1:
                                break
                        elif baseTransition == self.transitions.SHIFT:
                            if self.transitions.shift(conf) != -1:
                                break
                    else:
                        raise ValueError("The predicted transition is not recognized, expected errors")

            # Finish with operations build the dependency graph from Conf.arcs

            new_depgraph = copy.deepcopy(depgraph)
            for key in new_depgraph.nodes:
                node = new_depgraph.nodes[key]
                node['rel'] = ''
                # With the default, all the token depend on the Root
                node['head'] = 0
            for (head, rel, child) in conf.arcs:
                c_node = new_depgraph.nodes[child]
                c_node['head'] = head
                c_node['rel'] = rel
            result.append(new_depgraph)

        return result

    def save(self, filepath):
        """
        Save the parameters with pickle
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath):
        with open(filepath) as f:
            return pickle.load(f)
