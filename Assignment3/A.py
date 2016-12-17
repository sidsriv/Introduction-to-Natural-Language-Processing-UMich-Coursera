from main import replace_accented
from sklearn import svm
from sklearn import neighbors
import nltk
import collections


# don't change the window size
window_size = 10

# A.1
def build_s(data):
    '''
    Compute the context vector for each lexelt
    :param data: dic with the following structure:
        {
			lexelt: [(instance_id, left_context, head, right_context, sense_id), ...],
			...
        }
    :return: dic s with the following structure:
        {
			lexelt: [w1,w2,w3, ...],
			...
        }

    '''
    s = collections.defaultdict(set)

    # implement your code here
    for lexelt, instances in data.iteritems():
        for (instance_id, left, head, right, senseid) in instances:
            left_tokens = nltk.word_tokenize(left)
            right_tokens = nltk.word_tokenize(right)
            window = left_tokens[-window_size:] + right_tokens[:window_size]
            s[lexelt].update(window)

    # create an ordered list instead of a set for each lexelt because ordering is very important further
    for lexelt in s.keys():
        s[lexelt] = list(s[lexelt])
    return s


# A.1
def vectorize(data, s):
    '''
    :param data: list of instances for a given lexelt with the following structure:
        {
			[(instance_id, left_context, head, right_context, sense_id), ...]
        }
    :param s: list of words (features) for a given lexelt: [w1,w2,w3, ...]
    :return: vectors: A dictionary with the following structure
            { instance_id: [w_1 count, w_2 count, ...],
            ...
            }
            labels: A dictionary with the following structure
            { instance_id : sense_id }

    '''
    vectors = {}
    labels = {}

    # implement your code here
    for (instance_id, left_context, head, right_context, sense_id) in data:
        tokens = nltk.word_tokenize(left_context) + nltk.word_tokenize(right_context)
        vectors[instance_id] = [tokens.count(w) for w in s]
        labels[instance_id] = sense_id
    return vectors, labels


# A.2
def classify(X_train, X_test, y_train):
    '''
    Train two classifiers on (X_train, and y_train) then predict X_test labels

    :param X_train: A dictionary with the following structure
            { instance_id: [w_1 count, w_2 count, ...],
            ...
            }

    :param X_test: A dictionary with the following structure
            { instance_id: [w_1 count, w_2 count, ...],
            ...
            }

    :param y_train: A dictionary with the following structure
            { instance_id : sense_id }

    :return: svm_results: a list of tuples (instance_id, label) where labels are predicted by LinearSVC
             knn_results: a list of tuples (instance_id, label) where labels are predicted by KNeighborsClassifier
    '''

    svm_results = []
    knn_results = []

    svm_clf = svm.LinearSVC()
    knn_clf = neighbors.KNeighborsClassifier()

    # implement your code here
    x_matrix = []
    y_vector = []
    for instance_id, x in X_train.iteritems():
        x_matrix.append(x)
        y_vector.append(y_train[instance_id])

    svm_clf.fit(x_matrix, y_vector)
    knn_clf.fit(x_matrix, y_vector)
    for instance_id, x in X_test.iteritems():
        svm_results.append((instance_id, svm_clf.predict(x)[0]))
        knn_results.append((instance_id, knn_clf.predict(x)[0]))
    return svm_results, knn_results

# A.3, A.4 output
def print_results(results ,output_file):
    '''

    :param results: A dictionary with key = lexelt and value = a list of tuples (instance_id, label)
    :param output_file: file to write output

    '''

    # implement your code here
    # don't forget to remove the accent of characters using main.replace_accented(input_str)
    # you should sort results on instance_id before printing
    lines = []
    with open(output_file, 'w') as fp:
        for lexelt, predictions in results.iteritems():
            predictions.sort(key=lambda x: x[0])
            for instance_id, sense_id in predictions:
                lines.append((replace_accented(lexelt), replace_accented(instance_id),
                              replace_accented(unicode(sense_id))))
        lines.sort(key=lambda x: x[0])
        for line in lines:
            fp.write('{} {} {}\n'.format(*line))

# run part A
def run(train, test, language, knn_file, svm_file):
    s = build_s(train)
    svm_results = {}
    knn_results = {}
    for lexelt in s:
        X_train, y_train = vectorize(train[lexelt], s[lexelt])
        X_test, _ = vectorize(test[lexelt], s[lexelt])
        svm_results[lexelt], knn_results[lexelt] = classify(X_train, X_test, y_train)

    print_results(svm_results, svm_file)
    print_results(knn_results, knn_file)



