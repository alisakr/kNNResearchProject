'''
Created on Aug 8, 2014
@author Sean Massung
'''

import random

class LineDataset(object):

    def __init__(self, path=None):
        if path is None:
            return
        self.data = open(path).readlines()
        self.filenames = open(path + ".names").readlines()
        text_target = open(path + ".labels").readlines()
        self.target_names = list(set(text_target))
        self.target = []
        target_map = {}
        i = 0
        for target in self.target_names:
            target_map[target] = i
            i += 1
        for name in text_target:
            self.target.append(target_map[name])
        all_lists = list(zip(self.data, self.filenames, self.target))
        random.shuffle(all_lists)
        self.data, self.filenames, self.target = zip(*all_lists)

    def trainDocs(self):
        trainSet = LineDataset()
        mid = len(self.data)/2
        trainSet.data = self.data[0:mid]
        trainSet.filenames = self.filenames[0:mid]
        trainSet.target = self.target[0:mid]
        trainSet.target_names = self.target_names
        return trainSet

    def testDocs(self):
        testSet = LineDataset()
        mid = len(self.data)/2
        testSet.data = self.data[mid:]
        testSet.filenames = self.filenames[mid:]
        testSet.target = self.target[mid:]
        testSet.target_names = self.target_names
        return testSet
