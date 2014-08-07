'''
Created on Jul 28, 2014

@author: alisakr
'''

from trainingDoc import *
from tfidf import tfidf
import operator

class CenterQuery(object):
    '''
    classdocs
    '''

    def __init__(self, categoryName, numMostCommonWords):
        '''
        Constructor
        '''
        self.category = categoryName
        self.queryWordsMap = {}
        self.docCenterMap = {}
        self.reverseRankScores = None
        self.queryLength = 0
        self.sortedTF = []
        self.meanCategoryDocLen = 0.0
        self.numDocsAdded = 0
        self.numDocs = 0
        self.numCenterWords = numMostCommonWords
        self.combinedCenterSim = 0

    def setSortedTF(self):
        sortedTF = sorted(self.queryWordsMap.iteritems(), key=operator.itemgetter(1), reverse=True)
        self.sortedTF = sortedTF

    def createQuery(self, trainingDocs, meanDocLen, numDocs = "All"):
        i = 0
        noLimit = False
        if "All".startswith(str(numDocs)):
            noLimit = True
        for document in trainingDocs:
            if noLimit is False and i >=numDocs:
                break
            if str(self.category).startswith(trainingDocs[document].categoryName):
                self.addDocumentToCenterQuery(trainingDocs[document], meanDocLen)
                i+=1
        self.setSortedTF()

    def tooFewFilesOutput(self):
        output ="we couldn't add as many files to the center query as you wanted, we added "
        output = output + str(self.numDocs) + " files to the "
        output = output + str(self.category) + " category."
        return output

    def addDocumentToCenterQuery(self, document, meanDocLen):
        self.numDocs += 1
        for word in document.wordsMap:
            rawTF = document.wordsMap[word]
            docLen = document.docLength
            tf = float(rawTF)
            if self.queryWordsMap.has_key(word):
                self.queryWordsMap[word] += tf
            else:
                self.queryWordsMap[word] = tf
            self.queryLength += tf

    def compareDocToCenter(self, document, wordDFMap, numDocs, meanDocLen):
        if document.centerScore > 0.0:
            return
        if self.numCenterWords > self.queryLength:
            self.numCenterWords = self.queryLength
            tfIdf = tfidf(wordDFMap, numDocs, meanDocLen)
            document.centerScore = 0.0
            i = 0
            wordsUsed = 0
            while wordsUsed < self.numCenterWords:
                word = self.sortedTF[i][0]
                count = self.sortedTF[i][1]
                i += 1
                if tfIdf.isStopWord(word):
                    continue
                wordsUsed += 1
                if document.wordsMap.has_key(word):
                    document.centerScore += tfIdf.getTfIdf(count, word, document)
            self.combinedCenterSim += document.centerScore

    def averageCategoryCenterScore(self, averageCenterLength):
        sum = float(self.combinedCenterSim)
        return sum/float(self.numDocs)

    def setRankedScores(self):
        if self.reverseRankScores is not None:
            return
        self.reverseRankScores = {}
        list = sorted(self.docCenterMap.iteritems(), key=operator.itemgetter(1))
        i = 1
        for element in list:
            self.reverseRankScores[element[0]] = i
            i += 1

    def getDocRank(self, key):
        self.setRankedScores()
        return self.reverseRankScores[key]

    def getDocPercentile(self, key):
        docRank = self.getDocRank(key)
        percent = float(docRank)/float(self.numDocs)
        return percent*100.0
