'''
Created on 2017年6月10日

@author: Magister
'''
import time
from sklearn.svm import SVC
from predictor.kit.extractor import Extractor


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

class Classifier:
    @performance
    def preWorkForCLF(self, prop):
        ex = Extractor('E:\\pythonSpace\\BigDataMatch\\predictor\\kit\\dsjtzs_txfz_training.txt')
        ex.preProcess(prop)
    
        self.tr = ex.getTrainSet()
        self.trl = ex.getTrainLabel()
    @performance    
    def getClassifier(self):
        self.clf = SVC()
        self.clf.fit(self.tr, self.trl)
        
    @performance 
    def preWorkForPred(self,prop):
        ex = Extractor('E:\\pythonSpace\\BigDataMatch\\predictor\\kit\\dsjtzs_txfz_test1.txt')
        ex.preProcess(prop)
        
        self.te = ex.getTestSet()
        
    @performance 
    def predict(self):
        self.result = self.clf.predict(self.te)
    
    
        