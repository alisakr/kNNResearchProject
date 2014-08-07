'''
Created on Jul 25, 2014

@author: alisakr
'''
import string
import math
from Query import Query
from trainingDoc import trainingDoc


class tfidf(object):
    '''
    classdocs
    '''

    def __init__(self, wordDFMap, numDocs, meanDocLen, k1=1.6, beta=0.75):
        '''
        Constructor
        '''
        self.wordDF = wordDFMap
        self.numDocs = numDocs
        self.meanDocLen = meanDocLen
        self.k1 = k1
        self.beta = beta

    def getTfIdfs(self, word, trainDoc, queryTermsMap):
        idf = self.getIdf(word)
        tf = self.getBM25tf(word, trainDoc)
        return tf*idf*float(queryTermsMap[word])

    def isStopWord(self, word):
        if self.wordDF.has_key(word):
            D = float(self.wordDF[word])
            N = float(self.numDocs)
            DF = N - D
            DF /= N
            if DF < 0.10:
                return True
        return False

    def getTfIdf(self, queryTf, word, trainDoc):
        idf = self.getIdf(word)
        tf = float(trainDoc.wordsMap[word])
        docLen = float(trainDoc.docLength)
        tfIdfScore = idf*tf*self.meanDocLen/docLen
        if tfIdfScore <= 0.0:
            print "tfIdf Score on the word " + str(word) + " was zero."
        return tfIdfScore

    def getIdf(self, word):
        idf =  float(self.numDocs) + 1.0
        idfDenominator = float(self.wordDF[word]) + 1.0
        idf = idf/idfDenominator
        idf = math.log(idf)
        if idf < 0.0:
            print "idf was negative"
        return idf

    def adjustForDoclentgth(self, trainDoc, tfIdfScore):
        tfIdfScore = tfIdfScore / trainDoc.docLength
        tfIdfScore = tfIdfScore * self.meanDocLen
        return tfIdfScore

    def getBM25tfidf(self, word, trainDoc, queryTermsMap):
        tf = self.getBM25tf(word, trainDoc)
        idf = self.getBM25idf(word)
        tfIdfScore = tf*idf*float(queryTermsMap[word])
        if tfIdfScore <= 0.0:
            print "tfIdf Score on the word " + word + " was zero."
        return tfIdfScore

    def getBM25tf(self, word, trainDoc):
        rawTF = float(trainDoc.wordsMap[word])
        docLen = float(trainDoc.docLength)
        numerator = rawTF *(self.k1+1.0)
        denominator = rawTF + self.k1*((1.0-self.beta)+ (self.beta*docLen/self.meanDocLen))
        return numerator/denominator

    def getBM25idf(self, word):
        idf =  0.5 + float(self.numDocs) - float(self.wordDF[word])
        idfDenominator = float(self.wordDF[word]) + 0.5
        idf = idf/idfDenominator
        idf = math.log(idf)
        return idf
