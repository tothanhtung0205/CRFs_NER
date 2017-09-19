from io import open

arr = []
with open('data_vc2/data_nor.txt','r',encoding='utf-8') as f:
    for line in f:
       arr.append(line)

train_len = len(arr) - len(arr)/3
with open('data_vc2/train2.txt','w',encoding='utf-8') as f:
    for i in xrange(train_len):
        f.write(unicode(arr[i]))
    f.close()
with open('data_vc2/test2.txt','w',encoding='utf-8') as f:
    for i in xrange(train_len,len(arr)):
        f.write(unicode(arr[i]))
    f.close()
