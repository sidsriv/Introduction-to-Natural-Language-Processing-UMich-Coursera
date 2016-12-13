import A
import main
import unicodedata
import codecs
import sys

def build_dict(train):
    '''
    Count the frequency of each sense

    input: dict train: A dictionary with the following structure
        {
            lexelt: [(instance_id, left_context, head, right_context, sense_id), ...],
            ...
        }

    output: dict record: A dictionary with a lexelt as the key and the most frequent sense for that lexelt
            as the value
    '''
    data = {}
    for lexelt in train:
        data[lexelt] = {}
        inst_list = train[lexelt]
        for inst in inst_list:
            sense_id = inst[4]
            try:
                cnt = data[lexelt][sense_id]
            except KeyError:
                data[lexelt][sense_id] = 0
            data[lexelt][sense_id] += 1
    record = {}
    for key, cntDict in data.iteritems():
        sense = max(cntDict, key=lambda s: cntDict[s])
        record[key] = sense
    return record

def getFrequentSense(lexelt, sense_dict):
    '''
    Return the most frequent sense of a word (lexelt) in the training set
    '''
    sense = ''
    try:
        sense = sense_dict[lexelt]
    except KeyError:
        pass
    return sense

def most_frequent_sense(data, sense_dict, language):
    outfile = codecs.open(language + '.baseline', encoding='utf-8', mode='w')
    for lexelt, instances in sorted(data.iteritems(), key=lambda d: main.replace_accented(d[0].split('.')[0])):
        for instance in sorted(instances, key=lambda d: int(d[0].split('.')[-1])):
            instance_id = instance[0]
            sid = getFrequentSense(lexelt, sense_dict)
            outfile.write(main.replace_accented(lexelt + ' ' + instance_id + ' ' + sid + '\n'))
    outfile.close()

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print 'Usage: python baseline.py <language>'
        sys.exit(0)
    language = sys.argv[1]
    train_file = 'data/' + language + '-train.xml'
    dev_file = 'data/' + language + '-dev.xml'
    train = main.parse_data(train_file)
    test = main.parse_data(dev_file)
    sense_dict = build_dict(train)
    most_frequent_sense(test, sense_dict,language)
