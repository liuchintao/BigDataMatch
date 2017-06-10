'''
Created on 2017年6月5日

@author: Magister
'''
import random
import math
from builtins import int
from pip._vendor.pyparsing import line
class Extractor:
    '''
    Extract data from source file and convert them to feature vector.
    Parameters:
        self.__file
        self.trainList: list of list of coordination sets for train set.
        self.testList: list of list of coordination sets for test set.
        self.teIndex: list of index for test set.
        self.trLabel: list of label
        self.teLabel:
    '''
    
    def __init__(self, filePath):
        self.__loadfile(filePath)
    
    
    def __loadfile(self, filePath):
        f = open(filePath,'r')
        self.__file = f.readlines()
        f.close()
        
        
    def preProcess(self, prop):
        '''
        According to specific proportion, divide raw data to train set and test set.
        args:
            prop: the proportion of test set
        return:
            self.trainList: list of list of coordination sets for train set.
            self.testList: list of list of coordination sets for test set.
            self.teIndex: list of index for test set.
            self.trLabel: list of label
            self.teLabel:
        '''
        
        self.trainList = list()
        self.testList = list()
        self.teIndex = list()
        self.trLabel = list()
#         self.teLabel = list()
        
        self.teIndex = sorted(random.sample(range(1,len(self.__file) + 1), int(len(self.__file) * prop)))
        
        for line in self.__file:
            line = line.split()
            self.__singleLineProcessor(line)
        '''After loop, we get feature vector set'''
            

    def __initHist(self, start, end, num, dic):
        width = (end - start) / num
        #initial histogram
        for i in range(num + 1):
            dic[start + i * width] = 0
        return dic
    
    
    def __createDirHist(self, movement):
        '''
        Create direction sub feature vector
        The direction of vector is larger than 0 degree and lower than 90 degree
        args: 
            movement
        return:
            dirHist:{0,(0,30),[30,60),[60,90]}
        '''
        dirHist = dict()
        dirHist = self.__initHist(0, 90, 3, dirHist)
        i = 0
        while i < len(movement) - 1:
            vx = movement[i + 1][0] - movement[i][0]
            vy = movement[i + 1][1] - movement[i][1]
            if vx == 0:
                direction = 90
            else:
                direction = abs(math.atan(vy / vx) * 180 / math.pi)
            if direction == 0:
                dirHist[0] += 1
            elif direction / 30 < 3:
                dirHist[0 + 30 * int(direction / 30 + 1)] += 1
            else:
                dirHist[90] += 1
            i += 1
        return dirHist
    
    
    def __createAngDisHist(self, movement):
        '''
        PART I
        create angle of curvature metric list
        angle of curvature is the angle between vector AB and vector BC
        the range of angle of curvature vector is between 0 degree and 180 degree
        PART II
        create the curvature distance list
        Three point A, B, C create a triangle, and the height of triangle to AC
        is calculate by 2 * area(triangle) / len(AC).
        And the area = (s*(s-a)*(s-b)*(s-c)) ** 0.5,
        where s = a + b + c
        so the distance = 2 * area / len(AC) **2
        
        args:
            coordinate list that makes up mouse movement
        return:
            angHist: 0~180degree, and step is 30 degree.
            disHist: 0~20 and step is 1.
        '''
        angHist = dict()
        disHist = {-1 : 0, -2 : 0}
        angHist = self.__initHist(0, 180, 6, angHist)
        disHist = self.__initHist(0, 20, 20, disHist)
        
        if len(movement) <= 2:
            return angHist, disHist
        
        i = 0
        while i < len(movement) -2:
            ax = float(movement[i][0])
            ay = float(movement[i][1])
            bx = float(movement[i+1][0])
            by = float(movement[i+1][1])
            cx = float(movement[i+2][0])
            cy = float(movement[i+2][1])
            i += 1
            
            if (ax == bx and ay == by) or (bx == cx and by == cy):
                continue
            
            v1x = bx - ax
            v1y = by - ay
            v2x = cx - bx
            v2y = cy - by
            
            lenV1 = math.sqrt(v1x **2 + v1y **2)
            lenV2 = math.sqrt(v2x **2 + v2y **2)
            lenAC = math.sqrt((cx - ax) **2 + (cy - ay) **2)
            
            aCos = round((v1x * v2x + v1y * v2y) / (lenV1 * lenV2) , 6) 
            ang = math.acos(aCos) * 180 / math.pi
            
            if ang == 0:
                angHist[0] += 1
            elif ang / 30 < 6:
                angHist[0 + 30 * int(ang / 30 + 1)] += 1
            else:
                angHist[180] += 1
                
            if ang == 0:
                disHist[0] += 1
            if ang == 180:
                disHist[-1] = disHist.get(-1, 0) + 1
            else:
                s = lenV1 + lenV2 + lenAC
                area = math.sqrt(s * (s-lenV1) * (s-lenV2) * (s-lenAC))
                dist = round(2 * area / lenAC **2)
                if dist <= 20:
                    disHist[dist] += 1
                else:
                    disHist[-2] = disHist.get(-2, 0) + 1
                    
        return angHist, disHist
    
    
    def __createFeature(self, movement):
        '''
        Create feature vector consists of direction, angle of curvature and distance of curvature.
        args:
            movement:
        return:
            feaVector: the feature vector for a movement.
        '''
        feaVector = list()
        #get direction histogram
        dirHist = self.__createDirHist(movement)
        #get angle and distance of curvature histogram
        angHist, disHist = self.__createAngDisHist(movement)
        
        '''
        Now we get all feature sub_vector, and we would create feature vector.
        '''
#         print(dirHist)
#         print(angHist)
#         print(disHist)
        dirHist = sorted(dirHist.items(), key = lambda x : x[0])
        for dire in dirHist:
            feaVector.append(dire[1])
        angHist = sorted(angHist.items(), key = lambda x : x[0])
        disHist = sorted(disHist.items(), key = lambda x : x[0])
        for ang in angHist:
            feaVector.append(ang[1])
        for dis in disHist:
            feaVector.append(dis[1])
            
        return feaVector
    
    
    def __transMoveType(self, movement):
        move = list()
        for m in movement:
            m = m.split(",")
            x = float(m[0])
            y = float(m[1])
            t = float(m[2])
            c = tuple([x,y,t])
            move.append(c)
        return move
        
    
    def __singleLineProcessor(self, line):
        idx = int(line[0])
        if len(line) == 4:
            clazz = int(line[3])
        movement = line[1].split(";")
        movement.remove("")
        movement = self.__transMoveType(movement)
#         target = tuple(line[2].split(","))
        subVec = self.__createFeature(movement)
        if idx in self.teIndex:
            self.testList.append(subVec)
#             self.teLabel.append(clazz)
        else:
            self.trainList.append(subVec)
            self.trLabel.append(clazz)
        pass
    
    
    def getTestSet(self):
        return self.testList
    
    
    def getTestLabel(self):
        return self.teLabel
    
    
    def getTrainSet(self):
        return self.trainList
    
    
    def getTrainLabel(self):
        return self.trLabel