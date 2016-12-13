from dependencycorpusreader import DependencyCorpusReader
import os

BASE_PATH = '../data/'

def get_swedish_train_corpus():
    root = os.path.join(BASE_PATH, 'swedish/talbanken05')
    files = ['train/swedish_talbanken05_train.conll']
    return DependencyCorpusReader(root, files)

def get_swedish_test_corpus():
    root = os.path.join(BASE_PATH, 'swedish/talbanken05')
    files = ['test/swedish_talbanken05_test.conll']
    return DependencyCorpusReader(root, files)

def get_danish_train_corpus():
    root = os.path.join(BASE_PATH, 'danish/ddt')
    files = ['train/danish_ddt_train.conll']
    return DependencyCorpusReader(root, files)

def get_danish_test_corpus():
    root = os.path.join(BASE_PATH, 'danish/ddt')
    files = ['test/danish_ddt_test.conll']
    return DependencyCorpusReader(root, files)

def get_dutch_train_corpus():
    root = os.path.join(BASE_PATH, 'dutch/alpino')
    files = ['train/dutch_alpino_train.conll']
    return DependencyCorpusReader(root, files)

def get_dutch_test_corpus():
    root = os.path.join(BASE_PATH, 'dutch/alpino')
    files = ['test/dutch_alpino_test.conll']
    return DependencyCorpusReader(root, files)

def get_korean_train_corpus():
    root = os.path.join(BASE_PATH, 'korean')
    files = ['train/ko-universal-train.conll']
    return DependencyCorpusReader(root, files)

def get_korean_test_corpus():
    root = os.path.join(BASE_PATH, 'korean')
    files = ['test/ko-universal-test.conll']
    return DependencyCorpusReader(root, files)

def get_english_train_corpus():
    root = os.path.join(BASE_PATH, 'english')
    files = ['train/en-universal-train.conll']
    return DependencyCorpusReader(root, files)

def get_english_test_corpus():
    root = os.path.join(BASE_PATH, 'english')
    files = ['test/en-universal-test.conll']
    return DependencyCorpusReader(root, files)

def get_english_dev_corpus():
    root = os.path.join(BASE_PATH, 'english')
    files = ['dev/en-universal-dev.conll']
    return DependencyCorpusReader(root, files)
