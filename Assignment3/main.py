import sys
import unicodedata
from xml.dom import minidom
import A
import B

def replace_accented(input_str):
    nkfd_form = unicodedata.normalize('NFKD', input_str)
    return u"".join([c for c in nkfd_form if not unicodedata.combining(c)])

def parse_data(input_file):
    '''
	Parse the .xml data file (for both train and dev)
    :param str input_file: The input data file path
	:return dict: A dictionary with the following structure
		{
			lexelt: [(instance_id, left_context, head, right_context, sense_id), ...],
			...
		}
	'''
    xmldoc = minidom.parse(input_file)
    data = {}
    lex_list = xmldoc.getElementsByTagName('lexelt')
    for node in lex_list:
        lexelt = node.getAttribute('item')
        data[lexelt] = []
        inst_list = node.getElementsByTagName('instance')
        for inst in inst_list:
            instance_id = inst.getAttribute('id')
            l = inst.getElementsByTagName('context')[0]

            # For Spanish and Catalan
            try:
                l = l.getElementsByTagName('target')[0]
            except:
                pass

            left = l.childNodes[0].nodeValue.replace('\n', '').lower()
            head = l.childNodes[1].firstChild.nodeValue.replace('\n', '').lower()
            right = l.childNodes[2].nodeValue.replace('\n', '').lower()

            senseid = ''

            # if train then parse sense, if test then senseid = ''
            try:
                senseid = inst.getElementsByTagName('answer')[0].getAttribute('senseid')
                senseid = replace_accented(senseid).encode('ascii')
            except:
                senseid = ''
            data[lexelt].append((instance_id, left, head, right, senseid))

    return data


if __name__ == '__main__':
    if len(sys.argv) != 7:
        print 'Usage: python main.py <input_training file> <input test file> <output KNN file> <output SVM file> <output best file> <language>'
        sys.exit(0)

    train_file = sys.argv[1]
    test_file = sys.argv[2]
    knn_answer = sys.argv[3]
    svm_answer = sys.argv[4]
    best_answer = sys.argv[5]
    language = sys.argv[6]

    train_set = parse_data(train_file)
    test_set = parse_data(test_file)

    A.run(train_set, test_set, language, knn_answer, svm_answer)
    B.run(train_set, test_set, language, best_answer)