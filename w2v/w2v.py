# -*- coding: utf-8 -*-
from __builtin__ import classmethod
from _md5 import new

from nltk.util import pr

__author__ = ''
from gensim.models import Word2Vec
from io import open
import numpy as np
import math, os
from sklearn.cluster import KMeans,DBSCAN
from sklearn.externals import joblib


class features_extraction:
    def __init__(self):
        self.word_size = 200
        self.low = (-1) * math.sqrt(3.0/self.word_size)
        self.high = (-1) * self.low
        self.model = Word2Vec(min_count=5, negative=10, size=self.word_size, window=4, sg=1, iter=100, workers=4)
        self.sents = self.load_dataset('dataset/nomed_ner_vietnamese.txt')

    def load_dataset(self, dataset):
        print 'load dataset ...'
        sentences = []
        with open(dataset, 'r', encoding='utf-8') as f:
            sentence = []
            for w in f:
                w = w.rstrip(u'\n')
                if w == u'':
                    sentences.append(sentence)
                    sentence = []
                else:
                    word = w.split(u' ')[0].lower()
                    sentence.append(word)
        print "Load dataset completed"
        return sentences

    def run(self, dataset):
        try:
            self.model = Word2Vec.load('w2v_.pkl')
        except:
            self.model.build_vocab(self.sents)
            print 'training word2vec model ...'
            self.model.train(self.sents, total_examples=self.model.corpus_count, epochs=self.model.iter)
            self.model.save('w2v_.pkl')
            print 'training word2vec completed !!!'

    def cluster(self,dict_w2v):
        # try:
        #     word_centroid_map = joblib.load(dict_w2v)
        # except:
        print "Transfer to K-means input form..."
        list_word2vec = np.array([[]])
        list_word=self.model.wv.index2word
        for word in list_word:
            if list_word.index(word) == 0:
                list_word2vec = np.array([self.model[word]])
            else:
                x = self.model[word]
                x = np.array([x])
                list_word2vec = np.append(list_word2vec,x,axis=0)
        print "Training K-Means"
        kmeans = KMeans(500)
        idx = kmeans.fit_predict(list_word2vec)
        centroid = kmeans.cluster_centers_


        vector_cluster = zip(list_word2vec,self.model.wv.index2word,idx)
            #Print the cluster number
        new_vector_cluster = []
        final_vector_cluster = []
        for cluster in xrange(0,500):
            print "\nCluster %d" % cluster
            words = []
            words_raw = []
            for tup in vector_cluster:
                if tup[2] == cluster:   # if this word in clus i
                    words_raw.append(tup[1])
                    dist = np.linalg.norm(tup[0]-centroid[cluster])
                    if dist < 5:
                        words.append(tup[1])
                        new_vector_cluster.append(tup)
            print words

        for cluster in xrange(0,500):
            print "\nCluster %d" % cluster
            words_final = []
            for tup3 in new_vector_cluster:
                if tup3[2] == cluster:
                    words_final.append(tup3[1])
            if len(words_final) > 1:
                for tup4 in new_vector_cluster:
                    if tup4[2] == cluster:
                        final_vector_cluster.append(tup4)

        print final_vector_cluster
        for cluster in xrange(0, 500):
            print "\nCluster %d" % cluster
            word_fin = []
            for tup5 in final_vector_cluster:
                if tup5[2] == cluster:
                    word_fin.append(tup5[1])
            print word_fin
        return final_vector_cluster


    def get_word_vector(self, word):
        try:
            return list(self.model[word])
        except:
            # return np.zeros((self.word_size))
            return list(np.random.uniform(low=self.low, high=self.high, size=(self.word_size)))

    def is_allcaps(self, word):
        if word.isupper():
            return np.ones((1))
        else: return np.zeros((1))


    def is_init_cap(self, word):
        if word.istitle():
            return np.ones((1))
        else: return np.zeros((1))

    def is_lower(self, word):
        if word.islower():
            return np.ones((1))
        else: return np.zeros((1))

    def get_feature(self, word):
        v = self.get_word_vector(word)
        v = map(str,v)
        s = u'|'.join(v)
        return s


def get_cluster_dict():
    we = features_extraction()
    we.run('dataset/nomed_ner_vietnamese.txt')
    dict = we.cluster('idx_.pkl')
    return dict
dict = get_cluster_dict()
pass
def get_cluster(word):
    try:
        return dict[word]
    except:
        return -111
print get_cluster(u'ubnd')
print  get_cluster(u'xyzzz')








