'''
Created on 2017年6月10日

@author: Magister
'''
from predictor.kit.Extractor import Extractor
import numpy as np
from sklearn.svm import SVC
import time


def performance(f):
    def wrapper(*args):
        t1 = time.time()
        print(f.__name__, ' work start at: %s' %t1)
        r = f(*args)
        t2 = time.time()
        print(f.__name__, ' work stop at: %s' %t2)
        t = t2 - t1
        print(f.__name__, ' work spend %ss' %t)
        return r
    return wrapper

@performance
def getRealBlack(predTestl, tel):
    blackNum = 0
    black = 0
    jugBlack = 0
    i = 0
    while i < len(tel):
        if predTestl[i] == 0:
            jugBlack += 1
        if tel[i] != 0:
            i += 1
            continue
        black += 1
        if predTestl[i] == tel[i]:
            blackNum += 1
        i += 1
    return blackNum, black, jugBlack

@performance
def preWork(prop):
    ex = Extractor()
    ex.preProcess(prop)
    
    te = ex.getTestSet()
    tel = ex.getTestLabel()
    
    tr = ex.getTrainSet()
    trl = ex.getTrainLabel()
    
    return te, tel, tr, trl

@performance
def getClassifier(x, y):
    clf = SVC()
    clf.fit(x,y)
    return clf

@performance
def predict(clf, sample):
    rtnVal = clf.predict(sample)
    return rtnVal
    
if __name__ == '__main__':
    te, tel, tr, trl = preWork(0.3)
    
    x = np.array(tr)
    y = np.array(trl)
    
    clf = getClassifier(x, y)
    
    predTestl = predict(clf, te)
    
    realBlack, black, jugBlack, loc = getRealBlack(predTestl, tel)
    
    precision = realBlack / jugBlack
    recall = realBlack / black
    f = 5 * precision * recall / (2 * precision + 3 * recall) * 100
    
    print('realBlack in predict: ', realBlack)
    print('black in test: ', black)
    print('judge black is: ', jugBlack)
    
    print('Precision: ', precision)
    print('Recall: ', recall)
    print('final score: ', f)
    