'''
Created on Jul 27, 2014

@author: alisakr
'''
import scipy.special as sps
import numpy as np
import matplotlib.pyplot as plt
import collections
from collections import OrderedDict
import re
import nltk
import string
import random
import math
import sys
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from Query import Query
from nltk.stem.snowball import SnowballStemmer
from trainingDoc import trainingDoc
from tfidf import tfidf
import operator
from wordStemmer import *

class createCollection(object):
    '''
    classdocs
    '''

    def __init__(self, dataset):
        '''
        Constructor
        '''
        self.data = dataset.data
        self.filenames = dataset.filenames
        self.categories = dataset.target
        self.categoryNames = dataset.target_names
        self.trainDocs = {}
        self.wordDF = {}
        self.numDocs = 0
        self.flip = False
        self.meanDocLen = 0.0
        self.centersCalculated = False
        self.maxCenterScore = 0.0
        self.maxQueryDocSimilarity = 0.0
        self.setTrainingDocs()
        self.centroidWeightedDiff = 0
        self.centroidRight = 0
        self.weightedRight = 0
        self.harmonicMean = False
        self.averageCenterLen = 0.0
        self.centers = None
        self.centerMap = {}

    def setTrainingDocs(self):
        i = 0
        datasetLen = 0
        self.numDocs = 0
        for file in self.data:
            label = self.categories[i]
            words = file.split()
            fileWords = {}
            docLength = 0
            self.numDocs +=1
            for word in words:
                word = stemmedWord(word)
                if fileWords.has_key(word) is False:
                    fileWords[word] = 0
                    if self.wordDF.has_key(word):
                        self.wordDF[word] +=1
                    else:
                        self.wordDF[word] = 1
                fileWords[word] +=1
                docLength+=1
                datasetLen +=1
            document = trainingDoc(i, docLength, fileWords, label, self.categoryNames[label])
            self.trainDocs[i] = document
            i+=1
        self.meanDocLen = float(datasetLen)/float(self.numDocs)

    def simpleKNN(self, k, query, weighted=False):
        sortedScores = self.getDocSimilarityScores(query)
        self.getCategory(k, query, sortedScores, weighted)

    def getMaxCenterScore(self, k, sortedScores):
        i = 0
        maxScore = 0.0
        for key, value in sortedScores:
            if self.trainDocs[key].centerScore > maxScore:
                maxScore = self.trainDocs[key].centerScore
            i += 1
            if i >= k:
                return maxScore
    def getMaxQueryScore(self, sortedScores):
        for key, value in sortedScores:
            return value

    def getAverageCenterLen(self, centers):
        sum = 0.0
        numCenters = 0
        for center in centers:
            sum += centers[center].queryLength
            numCenters += 1
        averageCenterLen = sum/float(numCenters)
        self.averageCenterLen = averageCenterLen
        return averageCenterLen

    def neighborWeight(self, docCenterScore, docQueryScore, key, beta, selection = False, centerScores = []):
        docCenterScore *= beta
        docQueryScore *= (1.0-beta)
        docCenterScore *= self.averageCenterLen
        category = self.categoryNames[self.categories[key]]
        docCenterScore /= float(self.centers[category].queryLength)
        if selection:
            if centerScores.count(key) < 1:
                return -1.0
            if self.harmonicMean:
                return 1.0
            return docQueryScore + docCenterScore
        if docQueryScore == 0.0 and docCenterScore == 0.0:
            print "both doc center and doc query score are zero"
        elif docCenterScore == 0.0:
            print "doc center score is zero"
        sum = docCenterScore + docQueryScore + .000001
        if self.harmonicMean:
            score = 2.0*docCenterScore*docQueryScore/sum
            return score
        return sum

    def harmonicCenterScoreList(self, k, sortedScores):
        i = 0
        centerScores = {}
        queryScores = {}
        harmonicMeans = {}
        centerList = []
        minQuery = 0.000001
        maxQuery = minQuery
        maxCenterScore = 0.0
        for key, value in sortedScores:
            if i == 0:
                centerList.append(key)
                i += 1
                maxQuery += value
                continue
            if i >= (2*k -1):
                break
            queryScores[key] = value
            centerScores[key] = self.trainDocs[key].centerScore
            if centerScores[key] > maxCenterScore:
                maxCenterScore = centerScores[key]
            i += 1
        for key in centerScores:
            centerScores[key] = centerScores[key]/maxCenterScore
            queryScores[key] = (queryScores[key]+minQuery)/maxQuery
            harmonicMeans[key] = 2.0*centerScores[key]*queryScores[key]
            sum = centerScores[key] + queryScores[key]
            harmonicMeans[key] /= sum
        scores = sorted(harmonicMeans.iteritems(), key=operator.itemgetter(1), reverse=True)
        i = 1
        for docID, center in scores:
            if  i >= k:
                break
            centerList.append(docID)
            i += 1
        return centerList

    def centerScoreList(self, k, sortedScores):
        i = 0
        centerList = []
        maxQuery = 0.000001
        numAdded = 0
        for key, value in sortedScores:
            if numAdded >= k:
                break
            rank = self.centers[self.trainDocs[key].categoryName].getDocPercentile(key)
            cutOff = 100/(2*k)
            if self.flip:
                rank = 100.0 - rank
            if rank >= 25.0:
                if numAdded == 0:
                    maxQuery += value
                centerList.append(key)
                numAdded += 1
            i += 1
        return centerList

    def setCenterScores(self):
        if self.centersCalculated:
            return
        averageCenterLen = self.getAverageCenterLen(self.centers)
        self.updateCenterScores(self.centers, averageCenterLen)

    def centroidKNN(self, k, query, beta, selection=False):
        sortedScores = self.getDocSimilarityScores(query)
        averageCenterLen = self.getAverageCenterLen(self.centers)
        tags = {}
        i = 0
        maxCenterScore = self.getMaxCenterScore(k, sortedScores) + 0.000001
        if selection:
            limit = 2*k -1
            centerScoreList = self.centerScoreList(k, sortedScores)
            lastelement = centerScoreList[len(centerScoreList) -1]
        else:
            limit = k
            centerScoreList = None
        maxQuerySimilarity = 0.0
        for key, value in sortedScores:
            category = self.categories[key]
            if i >= k and selection is False:
                break
            if i == 0:
                maxQuerySimilarity = value + 0.000001
            docQueryScore = 100.0*((0.000001 + value)/maxQuerySimilarity)
            docCenterScore = 100.0*((self.trainDocs[key].centerScore + 0.000001)/maxCenterScore)
            neighborWeight = self.neighborWeight(docCenterScore, docQueryScore, key, beta, selection, centerScoreList)
            if tags.has_key(category):
                tags[category] +=  1.0 + neighborWeight
            else:
                tags[category] = 1.0 + neighborWeight
            i += 1
            if selection:
                if key == lastelement:
                    break
        query.inferredCategory = self.maxCategory(tags)

    def getDocSimilarityScores(self, query):
        similarityMap = {}
        for j in xrange(len(self.trainDocs)):
            similarity = self.getDocumentSimilarity(query.tfMap, self.trainDocs[j])
            similarityMap[j] = similarity
            if similarity > self.maxQueryDocSimilarity:
                self.maxQueryDocSimilarity = similarity
        sortedScores = sorted(similarityMap.iteritems(), key=operator.itemgetter(1), reverse=True)
        return sortedScores

    def getDocumentSimilarity(self, queryTermsMap, trainDoc):
        tfIdfHelper = tfidf(self.wordDF, self.numDocs, self.meanDocLen)
        similarity = 0.0
        for word in trainDoc.wordsMap:
            if queryTermsMap.has_key(word):
                similarity += tfIdfHelper.getTfIdf(queryTermsMap[word],word, trainDoc)
        return similarity

    def updateCenterScores(self, centers, averageCenterLen):
        if self.centersCalculated:
            return
        for j in xrange(len(self.trainDocs)):
            center = centers[self.trainDocs[j].categoryName]
            centerLen = center.queryLength
            center.compareDocToCenter(self.trainDocs[j], self.wordDF, self.numDocs, self.meanDocLen)
            #self.trainDocs[j].adjustCenterScore(centerLen, averageCenterLen)
            center.docCenterMap[j] = self.trainDocs[j].centerScore
            if self.trainDocs[j].centerScore > self.maxCenterScore:
                self.maxCenterScore = self.trainDocs[j].centerScore
        self.centersCalculated = True

    def getCategory(self, k, query, sortedScores, weighted):
        if weighted is False:
            self.unweightedCategory(k, query, sortedScores)
        else:
            self.simpleWeightedCategory(k, query, sortedScores)

    def unweightedCategory(self, k, query, sortedScores):
        tags = {}
        i = 0
        for key,value in sortedScores:
            if i >= k:
                break
            fileID = key
            category = self.categories[fileID]
            if tags.has_key(category):
                tags[category] +=1
            else:
                tags[category] = 1
            i += 1
        query.inferredCategory = self.maxCategory(tags)

    def maxCategory(self, categoryScores):
        maxScore = 0.0
        maxCategory = None
        for label in categoryScores:
            if float(categoryScores[label]) > maxScore:
                maxScore = float(categoryScores[label])
                maxCategory = label
        if maxCategory is None:
            return None
        return self.categoryNames[maxCategory]

    def simpleWeightedCategory(self, k, query, sortedScores):
        tags = {}
        i = 0
        for key,value in sortedScores:
            if i >= k:
                break
            category = self.categories[key]
            if value <= 0.0:
                if tags.has_key(category):
                    tags[category] +=  1.0
                else:
                    tags[category] = 1.0
                i += 1
                continue
            if tags.has_key(category):
                tags[category] += 1.0 + value*1000.0
            else:
                tags[category] = 1.0 + value*1000.0
            i += 1
        query.inferredCategory = self.maxCategory(tags)
