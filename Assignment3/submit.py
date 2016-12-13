import sys
import os

#==========================Read Learner's Infomation=============

learnerEmail = raw_input('Login (Email address): ')
learnerSecret = raw_input('One-time Password (from the assignment page. This is NOT your own account\'s password): ')

#==========================Assignment Word Sense Disambiguation=========

assignmentKey = '6li-kQHtEeaJEA6Jlpo7lQ'
partId1 = 'ttTM5'
partId2 = 'AIcBw'

partIdx = raw_input('Please enter which parts you want to submit: \n1: Part A\n2: Part B\nFor example, type "1 2" will submit part 1 and part 2\n')


#=========================Evaluation====================

import subprocess

def evaluate(files,test_files,baselines,references,scores):

  score_total = 0
  for i in range(len(files)):
    f = files[i]
    baseline = baselines[i]
    reference = references[i]
    test_file = test_files[i]
    score = scores[i]
    if not os.path.exists(f):
      print 'Please save your output file', f, 'under Assignment3 directory.'
      continue
      
    command = "./scorer2 " + f + " " + test_file
    print command
    
    #res = subprocess.check_output(command,shell = True)
    try:
      res = subprocess.check_output(command,shell = True)
    except Exception, e:
      res = None
      print 'scorer2 failed for',f
      sys.exit()

    #print res

    acc = 0
    if res:
      try:
        acc = float(res.split('\n')[2].split(' ')[2])
      except Exception, e:
        print 'scorer2 failed for',f
        sys.exit()

    print 'accuracy',acc,
    if acc < baseline:
      score_i = 0
    elif acc >= reference:
      score_i = score
    else:
      score_i = (score - score*(reference - acc)/(reference - baseline))

    score_total += score_i
    print 'score',score_i

  return score_total

def evaluate_part(partIdx):

  if partIdx == 1:
    files = ['KNN-English.answer','KNN-Spanish.answer','KNN-Catalan.answer','SVM-English.answer','SVM-Spanish.answer','SVM-Catalan.answer']
    #test_files = ['data/English-dev.key data/English.sensemap','data/Spanish-dev.key','data/Catalan-dev.key'] * 2
    test_files = ['data/English-dev.key','data/Spanish-dev.key','data/Catalan-dev.key'] * 2
    baselines = [0.535,0.684,0.678] * 2
    references = [0.550,0.690,0.705,0.605,0.785,0.805]
    scores = [10] * 6
    raw_score = evaluate(files,test_files,baselines,references,scores)
    return raw_score / 60.0
  elif partIdx == 2:
    files = ['Best-English.answer','Best-Spanish.answer','Best-Catalan.answer']
    #test_files = ['data/English-dev.key data/English.sensemap','data/Spanish-dev.key','data/Catalan-dev.key']
    test_files = ['data/English-dev.key','data/Spanish-dev.key','data/Catalan-dev.key']
    baselines = [0.605,0.785,0.805]
    references = [0.650,0.810,0.820]
    scores = [20,10,10]
    raw_score = evaluate(files,test_files,baselines,references,scores)
    return raw_score / 40.0


output1 = '0.0'
output2 = '0.0'
if '1' in partIdx:
    output1 = str(evaluate_part(1))
if '2' in partIdx:
    output2 = str(evaluate_part(2))

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
