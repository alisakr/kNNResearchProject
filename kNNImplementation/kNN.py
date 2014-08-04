'''
Created on Jul 25, 2014

@author: alisakr
'''
'''
Created on Jul 25, 2014

@author: alisakr
'''
import scipy.special as sps
import numpy as np
import matplotlib.pyplot as plt
import collections
import random 
import re
import optparse
import nltk
import string
import sklearn.datasets
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from collections import Counter
from dataCollection import dataCollection
from createCollection import createCollection
from Query import Query
from CenterQuery import *

def newCategory(categoryNum, numCategories):
    newCategory = random.randint(0, numCategories-2)
    if newCategory == categoryNum:
        newCategory = numCategories-1
    return newCategory

def contains_digits(d):
    _digits = re.compile('\d')
    return bool(_digits.search(d))

def remove_punctuation(s):
    exclude = set(string.punctuation)
    s = ''.join(ch for ch in s if ch not in exclude)
    return s

def cleaned_words(text):
    lowers = text.lower()
    #remove the punctuation using the character deletion step of translate
    no_punctuation = remove_punctuation(lowers)
    return no_punctuation

def get_tokens(text):
    no_punctuation = cleaned_words(text)
    tokens = nltk.word_tokenize(no_punctuation)
    return tokens


def extract_number(s):
    num = re.sub("[^0-9.]", " ", s)
    if num is None:
        print "no number found"
        return None
    if re.search("\.", num):
        return float(num)
    else:
        return int(num)

def getCenterQueries(categories, dataCollection, numDocs, numWords):
    centerQueries = {}
    for category in categories:
        centerQuery = CenterQuery(category, numWords)
        centerQueries[category] = centerQuery
        centerQuery.createQuery(dataCollection.trainDocs, dataCollection.meanDocLen, numDocs)
    return centerQueries

def getNumTests(testset, numTests):
    if "All".startswith(str(numTests)):
        numTests = len(testset.data)
    if numTests > len(testset.data):
        numTests = len(testset.data)
    numTests = int(numTests) 
    return numTests


def centroidKNN(testset, dataCollection, k, beta, numDocs, numWords, harmonicMean, selection, distributed, numTests= "All"):
    categories = dataCollection.categoryNames
    dataCollection.centersCalculated = False
    if harmonicMean:
        dataCollection.harmonicMean = True
    centerQueries = getCenterQueries(categories, dataCollection, numDocs, numWords)
    numTests = getNumTests(testset, numTests)
    queryList = getQueries(testset, numTests, distributed)
    numCorrect = 0
    dataCollection.centers = centerQueries
    dataCollection.setCenterScores()
    for i in xrange(numTests):
        query = queryList[i]
        dataCollection.centroidKNN(k, query, beta, selection)
        if query.inferredCategory is None:
            print "inferred category was none"
            continue
        if str(query.inferredCategory).startswith(query.category):
            numCorrect +=1
    #print "their were " + str(numTests) + " tests."
    accuracy = 100.0*float(numCorrect)/float(numTests)
    print str(accuracy)
    return accuracyByCategory(queryList, categories)
 
        
        

def testBasicKNN(testset, dataCollection, k,weighted, distributed, numTests = "All"):
        numTests = getNumTests(testset, numTests)
        queryList = getQueries(testset, numTests, distributed)
        numCorrect = 0
        categories = dataCollection.categoryNames
        for i in xrange(numTests):
            query = queryList[i]
            dataCollection.simpleKNN(k, query, weighted)
            if query.inferredCategory is None:
                print "inferred category was none"
                continue
            if str(query.inferredCategory).startswith(query.category):
                numCorrect +=1
        accuracy = 100.0*float(numCorrect)/float(numTests)
        print str(accuracy)
        return accuracyByCategory(queryList, categories)

def accuracyByCategory(queries, categoryNames):
    categoryMap = {}
    categoryNumRight = {}
    categoryNumQueries = {}
    for category in categoryNames:
        categoryMap[category] = 0.0
        categoryNumRight[category] = 0
        categoryNumQueries[category] = 0
    for query in queries:
        category = query.category
        categoryNumQueries[category] += 1
        if category.startswith(query.inferredCategory):
            if category.endswith(query.inferredCategory):
                categoryNumRight[category] += 1
    for category in categoryNames:
        categoryMap[category] = 100.0*float(categoryNumRight[category])/float(categoryNumQueries[category])
    return categoryMap

def randomizeTrainTags(trainingSet, percentRandomize):
    numCategories = len(trainingSet.target_names)
    for i in xrange(len(trainingSet.target)):
        modulus = i%100
        if modulus < percentRandomize:
            trainingSet.target[i] = newCategory(trainingSet.target[i], numCategories)
    return trainingSet

def makeRandom():
    randomizeAnswer = raw_input("randomize part of the training set (y/n) (default = no): " )
    randomizeAnswer = randomizeAnswer.lower()
    answer = yesOrNo(randomizeAnswer)  
    if answer is True:
        question = "percent of the training set to be randomized: "
        errorMessage = "invalid percent, type new percent randomized: "
        return int(getNumber(question, errorMessage))
    return 0
def getK():
    question = "integer value of k for all kNN: "
    errorMessage = "invalid k, give different value: "
    return int(getNumber(question, errorMessage))

def getHarmonicBeta():
    question = "beta for Harmonic Mean (0.0 < beta < 1.0, 0.5 is simple harmonic mean): "
    errorMessage = "invalid, give new beta:"
    return getNumber(question, errorMessage)

def getLinearInterpolationBeta():
    question = "beta for linear interpolation of cnkNN(0.0 <= beta <= 1.0): "
    errorMessage = "invalid, give new beta:"
    return getNumber(question, errorMessage)
    
def getQueries(testset, numQueries, evenDistribute=False):
    if evenDistribute:
        return getDistributedQueries(testset, numQueries)
    queryList = []
    if numQueries > len(testset.data):
        numQueries = len(testset.data)
    for i in xrange(numQueries):
        categoryNum = testset.target[i]
        file = testset.data[i]
        category = testset.target_names[categoryNum]
        query = Query(testset.filenames[i], file, category)
        queryList.append(query)
    return queryList

def getDistributedQueries(testset, numQueries):
    numCategories = len(testset.target_names)
    queriesPerTag = numQueries/numCategories
    i = 0
    j = 0 
    tagTestsMap = {}
    queryList = []
    while j < numQueries:
        if i > (len(testset.data) -1):
            print "your queries can not be evenly distributed"
            return None
        categoryNum = testset.target[i]
        file = testset.data[i]
        category = testset.target_names[categoryNum]
        if tagTestsMap.has_key(category):
            tagTests = tagTestsMap[category]
            if tagTests >= queriesPerTag:
                i += 1
                continue
            tagTestsMap[category] += 1
        else:
            tagTestsMap[category] = 1
        query = Query(testset.filenames[i], file, category)
        i += 1
        j += 1
        queryList.append(query)
    return queryList

    
        


def getNumber(userQuestion, errorMessage):
    input = raw_input(userQuestion)
    number = extract_number(input)
    while number is None:
        input = raw_input(errorMessage)
        number = extract_number(input)
    return number

def numQueries():
    question = "how many total test queries would you like to run?"
    errorMessage = "try again, how many total test queries would you like to run"
    return getNumber(question, errorMessage)
    
def use20newsgroups():
    answer = raw_input("would you like to use the default, 20newsgroups, dataset (y/n)?")
    return yesOrNo(answer)
    
def yesOrNo(answer):
    answer = answer.lower()
    if answer.find("y") != -1:
        return True
    return False
def evenDistribution():
    answer = raw_input("would you like test queries to be evenly distributed by category (y/n)?")
    return yesOrNo(answer)

    
if __name__ == '__main__':
    useNewsgroups= use20newsgroups()
    evenDistributed = evenDistribution()
    k = getK()
    numTests = numQueries()
    randomize = makeRandom()
    if useNewsgroups:
        trainingSet = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), shuffle = True, random_state=42)
        testSet = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'), shuffle = True, random_state = 42)
    else:
        print "we need you to provide a folder containing the training set, and another containing the test set"
        print "format of folder available at http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_files.html"
        trainSetFolder = raw_input("What is the folder containing the training set?")
        trainingSet = sklearn.datasets.load_files(trainSetFolder, shuffle = True, random_state = 42)
        testSetFolder = raw_input("What is the folder containing the test set?")
        testSet = sklearn.datasets.load_files(testSetFolder, shuffler = True, random_state=42)
    if randomize > 0:
        randomizeTrainTags(trainingSet, randomize)
    data  = createCollection(trainingSet)
    print "AT "+ str(randomize) +"% RANDOM and " + " K =" + str(k)
    print  " weighted KNN was " 
    wKNN = testBasicKNN(testSet, data, k, True, evenDistributed, numTests)
    print " weighted cnKNN was.. " 
    wcnKNN = centroidKNN(testSet, data, k, 0.0, 2000, 250, False, True, evenDistributed, numTests)
    print  " unweighted KNN was.. " 
    kNN = testBasicKNN(testSet, data, k, False, evenDistributed, numTests)
    print " unweighted cnKNN was.." 
    cnKNN = centroidKNN(testSet, data, k, 0.0, 2000, 250, True, True, evenDistributed, numTests)
    print "tag: kNN  cnKNN wKNN wcnKNN "    
    kNNs = 0
    wkNNs = 0
    cnkNNs = 0
    wcnkNNs = 0
    for category in wKNN:
        if kNN[category] > cnKNN[category]:
            kNNs +=1
        if kNN[category] < cnKNN[category]:
            cnkNNs +=1
        if wKNN[category] < wcnKNN[category]:
            wcnkNNs +=1
        if wKNN[category] > wcnKNN[category]:
            wkNNs +=1
        output = str(category) + ": "
        output += str(kNN[category]) + " "
        output += str(cnKNN[category]) + " "
        output += str(wKNN[category]) + " "
        output += str(wcnKNN[category]) + " " 
        print output
    explanation = "below comparison is the number of categories"
    explanation = explanation + " each scheme performed better than the scheme"
    explanation = explanation + " it's compared to, some categories might be tied"
    print explanation
    print "kNN v. cnKNN " + str(kNNs) + " v " + str(cnkNNs)
    print "wkNN v wcnKNN " + str(wkNNs) + " v " + str(wcnkNNs)
        
            
        
   


