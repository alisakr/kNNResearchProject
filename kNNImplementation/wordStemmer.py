'''
Created on Jul 28, 2014

@author: alisakr
'''

from nltk.stem.snowball import SnowballStemmer

def stemmedWord(word):
    stemmer = SnowballStemmer("english")
    word = ''.join(e for e in word if e.isalnum())
    word = word.lower()
    word = stemmer.stem(word)
    return word
