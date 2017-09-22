# -*- coding: utf-8 -*-

__author__ = ''

from gensim.models import Word2Vec
from io import open
import numpy as np
import math, os
from sklearn.cluster import KMeans



class features_extraction:
    def __init__(self):
        self.word_size = 200
        self.low = (-1) * math.sqrt(3.0/self.word_size)
        self.high = (-1) * self.low
        self.model = Word2Vec(min_count=5, negative=10, size=self.word_size, window=4, sg=1, iter=100, workers=4)



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
        return sentences


    def run(self, dataset):
        try:
            self.model = Word2Vec.load('w2v.pkl')
        except:
            sentences = self.load_dataset(dataset)
            self.model.build_vocab(sentences)
            print 'training word2vec model ...'
            self.model.train(sentences, total_examples=self.model.corpus_count, epochs=self.model.iter)
            self.model.save('w2v.pkl')
            print 'training word2vec completed !!!'

    def cluster(self,dataset):
        sents = self.load_dataset(dataset)
        list_word2vec = []
        for sent in sents:
            for word in sent:
                word2vec = self.get_word_vector(word)
                list_word2vec.append(word2vec)
        #list_word2vec = list(set(list_word2vec))

        kmeans = KMeans(100)

        idx = kmeans.fit_predict(list_word2vec)

        model = self.model
        word_centroid_map = dict(zip(model.wv.index2word, idx))

        for cluster in xrange(0, 10):

            # Print the cluster number

            print "\nCluster %d" % cluster

            # Find all of the words for that cluster number, and print them out

            words = []

            for i in xrange(0, len(word_centroid_map.values())):

                if (word_centroid_map.values()[i] == cluster):
                    words.append(word_centroid_map.keys()[i])

        print words

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


if __name__ == '__main__':
    aaa = features_extraction()
    aaa.cluster('dataset/nomed_ner_vietnamese.txt')


    we = features_extraction()
    we.run('dataset/nomed_ner_vietnamese.txt')
    x = we.model.most_similar(u'viện_kiểm_sát', topn=5)
    print(x)
    yyy = we.model.most_similar(u'công_ty', topn=5)
    print(yyy)
    zzz = we.model.most_similar(u'ubnd', topn=5)
    x_w2v  = we.get_word_vector(u'ngân_hàng')
    y_w2v = we.get_word_vector(u'nợ')

    ttt = [sum(a) for a in zip(x_w2v,y_w2v)]
    print ttt







