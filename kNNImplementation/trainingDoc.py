'''
Created on Jul 26, 2014

@author: alisakr
'''



class trainingDoc(object):
    '''
    classdocs
    '''


    def __init__(self, docID, docLength, wordsMap, category, categoryName):
        '''
        Constructor
        '''
        self.docID = docID
        self.docLength = docLength
        self.wordsMap = wordsMap
        self.category = category
        self.categoryName = categoryName
        self.centerScore = 0.0
    
    def adjustCenterScore(self, centerLength, averageCenterLen):
        self.centerScore = self.centerScore*averageCenterLen/float(centerLength) 
        