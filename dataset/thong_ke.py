# -*- coding=utf-8 -*-
from io import open
def read_file(file_name):
    sents = []
    sequence = []
    with open(file_name, 'r', encoding='utf-8') as f:
        for line in f:
            if line == '\n':
                sents.append(sequence)
                sequence = []
            else:
                line = line.encode('utf-8')
                word_pos_label = tuple(filter(None, line.split(' ')))
                sequence.append(word_pos_label)
    return sents

sents = read_file('nomed_ner_vietnamese.txt')
pass
b_org = []
prev_b_org = []
for sent in sents:
    for i in range(0,len(sent)):
        if(sent[i][2]) == "B-PER\n":
            b_org.append(unicode(sent[i][0],'utf-8').lower())
            try:
                x = unicode(sent[i-1][0],'utf-8').lower()
            except:
                x = "BOS"
            prev_b_org.append(x)
b_org.sort()
prev_b_org.sort()
import collections

def write_file(file_name,arr):
    counter = collections.Counter(arr)
    common = counter.most_common(200)
    pass
    with open(file_name,'w+',encoding='utf-8') as f:
        for word in common:
            x = u" ".join([word[0],str(word[1])])
            f.write(x)
            f.write(u'\n')
        f.close()

write_file('thong_ke/prev_b_per.txt',prev_b_org)


