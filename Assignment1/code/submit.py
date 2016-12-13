import sys
import os

#==========================Read Learner's Infomation=============

learnerEmail = raw_input('Login (Email address): ')
learnerSecret = raw_input('One-time Password (from the assignment page. This is NOT your own account\'s password): ')

#==========================Assignment Dependency parsing=========
from providedcode import dataset
from providedcode.transitionparser import TransitionParser
from providedcode.evaluate import DependencyEvaluator
from featureextractor import FeatureExtractor
from transition import Transition

assignmentKey = '5GeYTwHtEeaB7Ar4zuSs2w'
partId1 = 'iVBr2'
partId2 = 'RBXV2'
partId3 = '9hMr6'

partIdx = raw_input('Please enter which parts you want to submit: \n1: English\n2: Danish\n3: Swedish\nFor example, type "2 3" will submit part 2 and part 3\n')


#=========================Evaluation====================

def evaluate_parse(partIdx):
  if partIdx == 3:
    print 'Evaluating your swedish model ... '
    testdata = dataset.get_swedish_test_corpus().parsed_sents()
    if not os.path.exists('./swedish.model'):
      print 'No model. Please save your model as swedish.model at current directory before submission.'
      sys.exit(0)
    tp = TransitionParser.load('swedish.model')
    parsed = tp.parse(testdata)
    ev = DependencyEvaluator(testdata, parsed)
    uas, las = ev.eval()
    print 'UAS:',uas
    print 'LAS:',las
    swed_score = (min(las, 0.7) / 0.7) ** 2
    return swed_score
  
  if partIdx == 1:
    print 'Evaluating your english model ... '
    testdata = dataset.get_english_test_corpus().parsed_sents()
    if not os.path.exists('./english.model'):
      print 'No model. Please save your model as english.model at current directory before submission.'
      sys.exit(0)
    tp = TransitionParser.load('english.model')
    parsed = tp.parse(testdata)
    ev = DependencyEvaluator(testdata, parsed)
    uas, las = ev.eval()
    print 'UAS:',uas
    print 'LAS:',las
    eng_score = (min(las, 0.7) / 0.7) ** 2
    return eng_score
  
  if partIdx == 2:
    print 'Evaluating your danish model ... '
    testdata = dataset.get_danish_test_corpus().parsed_sents()
    if not os.path.exists('./danish.model'):
      print 'No model. Please save your model danish.model at current directory before submission.'
      sys.exit(0)
    tp = TransitionParser.load('danish.model')
    parsed = tp.parse(testdata)
    ev = DependencyEvaluator(testdata, parsed)
    uas, las = ev.eval()
    print 'UAS:',uas
    print 'LAS:',las
    dan_score = (min(las, 0.7) / 0.7) ** 2
    return dan_score

output1 = '0.0'
output2 = '0.0'
output3 = '0.0'
if '1' in partIdx:
    output1 = str(evaluate_parse(1))
if '2' in partIdx:
    output2 = str(evaluate_parse(2))
if '3' in partIdx:
    output3 = str(evaluate_parse(3))

#======================Submit Score========================

cmd = 'curl -X POST -H "Cache-Control: no-cache" -H "Content-Type: application/json" -d '

url = 'https://www.coursera.org/api/onDemandProgrammingScriptSubmissions.v1'
data = {  
      "assignmentKey": assignmentKey,  
      "submitterEmail": learnerEmail,  
      "secret": learnerSecret,  
      "parts": {  
        partId1: {  
          "output": output1
        },  
        partId2: {
          "output": output2
        },
        partId3: {
          "output": output3
        }
      }  
    }
'''
if partIdx == 1:
    data = {  
      "assignmentKey": assignmentKey,  
      "submitterEmail": learnerEmail,  
      "secret": learnerSecret,  
      "parts": {  
        partId1: {  
          "output": output
        },  
        partId2: {
        },
        partId3: {
        }
      }  
    }
elif partIdx == 2:
    data = {  
      "assignmentKey": assignmentKey,  
      "submitterEmail": learnerEmail,  
      "secret": learnerSecret,  
      "parts": {  
        partId1: {  
        },  
        partId2: {
          "output": output
        },
        partId3: {
        }
      }  
    }
elif partIdx == 3:
    data = {  
      "assignmentKey": assignmentKey,  
      "submitterEmail": learnerEmail,  
      "secret": learnerSecret,  
      "parts": {  
        partId1: {  
        },  
        partId2: {
        },
        partId3: {
          "output": output
        }
      }  
    }
else:
    print 'Invalid partID'
    sys.exit()
'''
curlcmd = cmd + "'" + str(data).replace("'",'"') +  "'" + " '" + url + "'"
print curlcmd
print
os.system(curlcmd)
