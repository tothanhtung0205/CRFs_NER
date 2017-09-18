# -*- coding: utf-8 -*-


import gensim
import sys


reload(sys)
sys.setdefaultencoding('utf-8')

sentences = [u"the quick brown fox jumps over the lazy dogs", u"Then a cop quizzed Mick Jagger's ex-wives briefly." ]
model = gensim.models.Word2Vec(min_count=1, size=2, window=2, sg=1, iter=10)
model.build_vocab([s.encode('utf-8').split() for s in sentences])
sentences = [s.encode('utf-8').split() for s in sentences]
model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)
print(model[u'jumps'])
model.build_vocab([[u'zzz']], update=True)
model.train([[u'zzz']], total_examples=model.corpus_count, epochs=model.iter)
print(model[u'jumps'])
print(model[u'zzz'])
pass