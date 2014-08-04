'''
Created on Jul 26, 2014

@author: alisakr
'''
import scipy.special as sps
import numpy as np
import matplotlib.pyplot as plt
import collections 
import re
import nltk
import string
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from collections import Counter
from trainingDoc import trainingDoc


class dataCollection(object):
    '''
    classdocs
    '''


    def __init__(self, dataset):
        self.trainDocs = {}
        self.vectorizer = CountVectorizer()
        self.dataset = dataset
        self.wordsList = None
        self.docVectors = None
        self.numDocs = 0 
        self.wordsToNumber = {}
        self.setWords()
        self.meanDocLen = 0.0
        
    
    def setWords(self):
        self.docVectors = self.vectorizer.fit_transform(self.dataset.data)
        tf_transformer = TfidfTransformer(use_idf=False).fit(self.docVectors)
        X_train_tf = tf_transformer.transform(self.docVectors)
        for piece in X_train_tf:
            print piece
        self.wordsList = self.vectorizer.get_feature_names()
        self.createWordNumberMap()
        self.createTrainDocs()
    
    def createWordNumberMap(self):
        i = 0 
        for word in self.wordsList:
            self.wordsToNumber[word] = i
            i +=1
    
    def findMostSimilarDocuments(self, text):
        return None
    
    def trainDocTest(self):
        self.IDs()
        
    def extract_number(self, s):
        num = re.sub("[^0-9.]", " ", s)
        if re.search("\.", num):
            return float(num)
        else:
            if re.search("[0-9]", num) is None:
                return None
            return int(num)
    
    def IDs(self):
        list = str(self.docVectors).splitlines()
        i = 0 
        for node in list:
            if i > 10:
                break
            #print node
            words = str(node).split()
            doc = self.extract_number(words[0])
            termID = self.extract_number(words[1])
            freq = self.extract_number(words[2])
            output = "words are "
            output2 = "ids are " + str(doc) + " "
            output2 = output2 + str(termID) + " "
            output2 = output2 + str(freq)
            for word in words:
                output = output + word + " "
            print output
            print output2
            i += 1 
    
    def createTrainDocs(self):
        docTermFreqs = str(self.docVectors).splitlines()
        prevDoc = 0
        currentDocLen = 0
        sumDocLens = 0 
        termsMap = {}
        for elem in docTermFreqs:
            valueList = self.getTrainDocData(elem)
            if valueList is None:
                continue
            docID = valueList[0]
            termID = valueList[1]
            freq = valueList[2]
            sumDocLens += freq
            if docID == prevDoc:
                currentDocLen += freq
            else:
                document = trainingDoc(docID, currentDocLen, termsMap)
                print "document has " + str(currentDocLen) + " words"
                self.trainDocs[docID] = document
                prevDoc = docID
                termsMap = {}
                currentDocLen = freq
            termsMap[termID] = freq
        self.trainDocs[docID] = trainingDoc(docID, currentDocLen, termsMap)
        self.numDocs = len(self.trainDocs.keys())
        self.meanDocLen = float(sumDocLens)/float(self.numDocs)
        print "mean Doc Length is " + str(self.meanDocLen)
    
    def getTrainDocData(self, docTermFreq):
        words = str(docTermFreq).split()
        docID = self.extract_number(words[0])
        if docID is None:
            return None
        termID = self.extract_number(words[1])
        freq = self.extract_number(words[2])
        return [docID, termID, freq]
    def testFunction(self):
        return None
             