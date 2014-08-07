'''
Created on Jul 26, 2014

@author: alisakr
'''

from wordStemmer import *

class Query(object):
    def __init__(self, name, text, category):
        self.docName = name
        self.text = text
        self.wordCount = 0
        self.uniqueWords = 0
        self.tfMap = {}
        self.category = category
        self.inferredCategory = None
        self.addWordsFromText()

    def addWord(self, word):
        word = stemmedWord(word)
        if self.containsTerm(word):
            self.tfMap[word] +=1
        else:
            self.tfMap[word] = 1
            self.uniqueWords +=1
        self.wordCount +=1

    def containsTerm(self, word):
        for key in self.tfMap.keys():
            if key is word:
                return True
        return False

    def addWords(self, words):
        for word in words:
            self.addWord(word)

    def addWordsFromText(self):
        words = self.text.split()
        self.addWords(words)
