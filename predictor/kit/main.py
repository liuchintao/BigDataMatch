'''
Created on 2017年6月10日

@author: Magister
'''
from predictor.kit.Classifier import Classifier

if __name__ == '__main__':
    clfer = Classifier()
    clfer.preWorkForCLF(0)
    clfer.getClassifier()
    clfer.preWorkForPred(1)
    clfer.predict()
    print('result length is: ', len(clfer.result))
    loc = [i+1 for i in range(len(clfer.result)) if clfer.result[i] == 0]
    print('Amount of black: ', len(loc))
    f = open(r'E:\pythonSpace\BigDataMatch\predictor\kit\test.txt', 'w')
    for i in loc:
        wline = str(i) + '\n'
        f.write(wline)
    f.close()