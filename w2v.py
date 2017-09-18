# -*- coding: utf-8 -*-

__author__ = 'nobita'

from gensim.models import Word2Vec
from io import open
import numpy as np
import math, os




class features_extraction:
    def __init__(self):
        self.word_size = 15
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
    we = features_extraction()
    we.run('dataset/nojmed_ner_vietnamese.txt')
    xxx = we.model.most_similar(u'viện_kiểm_sát', topn=5)
    yyy = we.model.most_similar(u'công_ty', topn=5)
    zzz = we.model.most_similar(u'ubnd', topn=5)
    print we.get_feature(u'ngân_hàng')
    print we.get_feature(u'nợ')



