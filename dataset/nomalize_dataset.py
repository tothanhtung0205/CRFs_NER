# coding=utf-8
from pyvi.pyvi import ViPosTagger,ViTokenizer
from io import open

dataset_arr = []
with open('raw_data_vc_small.txt', 'r', encoding='utf-8') as f:
    for line in f:
        if line == '\n':
            word_pos_label = ['\n','\n','\n']
        else:
            word_pos_label = line.split(' ')
        dataset_arr.append(word_pos_label)


nomed_arr = []
k=0
i=0
word = []
while i < len(dataset_arr):

        temp = ""
        if(len(dataset_arr[i]) == 3 and dataset_arr[i][2]=='B-PER\n'):
            temp += dataset_arr[i][0]
            pos = dataset_arr[i][1]
            label = dataset_arr[i][2]

            word = dataset_arr[i+1]
            if(len(word) == 3 and word[2]=='I-PER\n'):
                temp =  temp + "_" + dataset_arr[i+1][0]
                if(dataset_arr[i+2][2]=='I-PER\n'):
                    temp = temp + "_" + dataset_arr[i+2][0]
                    if (dataset_arr[i + 3][2] == 'I-PER\n'):
                        temp = temp + "_" + dataset_arr[i + 3][0]

            nomed_arr.append([temp,pos,label])
            i+=1

        elif ( len(dataset_arr[i]) == 3 and dataset_arr[i][2]=='I-PER\n'):
            i+=1
        else:
            nomed_arr.append(dataset_arr[i])
            i+=1
        pass
with open('nomed_ner_vietnamese.txt', 'w', encoding='utf-8') as f:
    lai = ""
    for word in nomed_arr:
        if word[1]=='\n':
            lai = '\n'
        else:
            lai = " ".join(word)
        f.write(unicode(lai))
    f.close()








