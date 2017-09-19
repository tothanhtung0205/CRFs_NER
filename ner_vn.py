# -*- coding=utf-8 -*-
from io import open
import sklearn_crfsuite
from sklearn_crfsuite import metrics
from sklearn.metrics import make_scorer
import scipy.stats
from pyvi.pyvi import ViPosTagger,ViTokenizer
from sklearn.externals import joblib
from sklearn.model_selection import RandomizedSearchCV
from regex import get_regex
import random
from w2v import features_extraction
from collections import Counter


w2v_model = features_extraction()
w2v_model.run('dataset/nomed_ner_vietnamese.txt')

#Featurelize sentences
class Shape:

    @staticmethod
    def contains_digit(str):
        for char in str:
            if char.isdigit():
                return True
        return False

    @staticmethod
    def itwp(str):
        if str.istitle() and '.' in str:
            return True
        else:
            return False

    @staticmethod
    def is_full_name(word):
        # To_Thanh_Tung return true
        # To_thanh_tung return false

        if '_' not in word:
            return  False

        temp = word.split('_')
        for token in temp:
            if token.istitle() == False:
                return False
        return True

# def get_shape(word):
#     shape = ""
#     if word.isdigit():      # 123,33,52123
#         shape = "so"
#     elif contains_digit(word):      #a12,15B,2231XXX
#         shape = "ma"
#     elif word.isupper():            #UBND,CLGT,...
#         shape = "viet hoa"
#     elif is_full_name(word):        #To_Thanh_Tung
#         shape = "ten day du"
#     elif word.istitle():            #Nam,An,Huy.....
#         shape = 'title'
#     elif word.islower():            #abc,xyz
#         shape = 'viet thuong'
#     else:
#         shape = 'other'             #mEo,iPhone
#     return shape

def single_features(sent, i):
    raw_word_0 = sent[i][0]
    word_0 = sent[i][0].lower()
    postag_0 = sent[i][1]

    word_minus_1 = sent[i-1][0].lower() if i>0 else "BOS"
    postag_minus_1 = sent[i-1][1] if i>0 else "BOS"
    R_minus_1 = get_regex(sent[i-1][0].lower()) if i>0 else "BOS"

    word_minus_2 = sent[i-2][0].lower() if i>1 else "BOS"
    postag_minus_2 = sent[i-2][1] if i>1 else "BOS"

    word_add_1 = sent[i+1][0].lower() if i<len(sent)-1 else "EOS"
    postag_add_1 = sent[i+1][1] if i < len(sent)-1 else "EOS"
    R_add_1 = get_regex(sent[i+1][0].lower()) if i < len(sent)-1 else "EOS"

    word_add_2 = sent[i + 2][0].lower() if i < len(sent)-2  else "EOS"
    postag_add_2 = sent[i + 2][1] if i < len(sent)-2 else "EOS"

    R_0 = get_regex(word_0)


    w2v = w2v_model.get_word_vector(word_0)
    features = {
        # co the them chunk va regular express
        'bias': 1.0,
        'W(0)': word_0,  # W_0,
        'P(0)': postag_0,  # P_0
        'R(0)':R_0,


        #shape of word
        'is_digit':raw_word_0.isdigit(),
        'contain_digit':Shape.contains_digit(raw_word_0),
        'is_upper':raw_word_0.isupper(),
        'W(0)+is_upper':raw_word_0+'+'+str(raw_word_0.isupper()),
        'P(0)+is_upper'
        
        'is_title':raw_word_0.istitle(),
        'is_lower':raw_word_0.islower(),
        'is_title_with_period':Shape.itwp(raw_word_0),
        'is_fullname':Shape.is_full_name(raw_word_0),

        'L1(0)': word_0.count('_'),  # bao nhieu tieng trong tu
        'L2(0)': len(word_0),



        'W(-1)':word_minus_1,
        'P(-1)':postag_minus_1,
        'R(-1)':R_minus_1,

        'W(-2)':word_minus_2,
        'P(-2)':postag_minus_2,

        'W(+1)':word_add_1,
        'P(+1)':postag_add_1,
        'R(+1)':R_add_1,

        'W(+2)':word_add_2,
        'P(+2)':postag_add_2,


        'R(-1)+R(0)':R_minus_1+'+'+R_0,
        'R(0)+R(1)':R_0+'+'+R_add_1,

        'W(-1)+W(0)':word_minus_1+"+"+word_0,
        'W(0)+W(1)' :word_0+"+"+word_add_1,


        'P(-1)+P(0)':postag_minus_1+'+'+postag_0,
        'P(0)+P(1)':postag_0+'+' +postag_add_1,

        'W(0)+P(0)':word_0+'+'+postag_0,
        'W(0)+P(1)':word_0+'+'+postag_add_1,
        'W(0)+P(-1)': word_0 + '+' + postag_minus_1,
        # 'W(0)+O(0)':word_0 + '+' + O_0,
    }
    return features


def word2features(sent, i):
    features = {}
    features.update(single_features(sent,i))
    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    arr = []
    for tup in sent:
        try:
            arr.append(tup[2])
        except:
            arr.append('O\n')
    return arr

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


# Reading file....
train_sents = read_file('dataset/data_vc2/train.txt')
print "featuring sentence..."
X_train = [sent2features(s) for s in train_sents]
y_train = [sent2labels(s) for s in train_sents]


def fit(model):
    #training phase
    try:
        crf = joblib.load(model)
        print 'load model completed !!!'
        return crf
    except: crf = None
    c2_rs =  0.1
    c1_rs = 0.1
    if crf == None:
        crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=c1_rs,
            c2=c2_rs,
            max_iterations=100,
            all_possible_transitions=True,
        )

        # featulize test set

        crf.fit(X_train, y_train)
        joblib.dump(crf, model)
        return crf
        #estimate model...
def optimize(model):
    crf = joblib.load(model)
    #rs_x_train = X_train[:len(X_train) / 10]
    #rs_y_train = y_train[:len(y_train) / 10]

    rs_x_train = random.sample(X_train,len(X_train/10))
    rs_y_train = random.sample(y_train, len(y_train / 10))

    labels = list(crf.classes_)
    params_space = {
        'c1': scipy.stats.expon(scale=0.5),
        'c2': scipy.stats.expon(scale=0.05),
    }

    # use the same metric for evaluation
    f1_scorer = make_scorer(metrics.flat_f1_score,
                            average='weighted', labels=labels)

    # search
    rs = RandomizedSearchCV(crf, params_space,
                            cv=3,
                            verbose=1,
                            n_jobs=-1,
                            n_iter=50,
                            scoring=f1_scorer)
    rs.fit(rs_x_train, rs_y_train)
    print('best params:', rs.best_params_)
    print('best CV score:', rs.best_score_)

def print_state_features(state_features):
    for (attr, label), weight in state_features:
        print("%0.6f %-8s %s" % (weight, label, attr))



def estimate(model):
    crf = joblib.load(model)


    #scr = crf.score(X_train,y_train)
    #print "train_score : %f" %scr


    test_sents = read_file('dataset/data_vc2/test.txt')

    X_test = [sent2features(s) for s in test_sents]
    y_test = [sent2labels(s) for s in test_sents]
    #print "test_score: %f" %(crf.score(X_test,y_test))

    labels = list(crf.classes_)
    # for lb in ['O\n','I-PRO\n','B-TOUR\n','I-TOUR\n','O','B-PER','B-LOC','B-ORG']:
    #     labels.remove(lb)
    labels = ['B-PER\n','I-PER\n','B-LOC\n','I-LOC\n','B-ORG\n','I-ORG\n']
    y_pred = crf.predict(X_test)



    kq = metrics.flat_f1_score(y_test, y_pred,
                          average='weighted', labels=labels)
    print kq

    # group B and I results
    sorted_labels = sorted(
        labels,
        key=lambda name: (name[1:], name[0])
    )
    print(metrics.flat_classification_report(
        y_test, y_pred, labels=sorted_labels, digits=3
    ))







#test a sentences
def test_ner(crf, test_sent):
    arr_featurized_sent = []
    postaged_sent = ViPosTagger.postagging(ViTokenizer.tokenize(test_sent))
    #postaged_sent = ViPosTagger.postagging(test_sent)

    print postaged_sent
    test_arr = []
    for i in xrange(len(postaged_sent[0])):
        test_arr.append((postaged_sent[0][i], postaged_sent[1][i]))
    print test_arr
    featurized_sent = sent2features(test_arr)
    arr_featurized_sent.append(featurized_sent)
    predict = crf.predict(arr_featurized_sent)
    return zip(test_arr,predict[0])


def predict(crf, query):

    query = unicode(query, encoding='utf-8')
    kqcc = test_ner(crf, query)
    x = [x[0][0] + u' -- ' + unicode(x[1], 'utf-8') for x in kqcc]
    s2 =   u'\n'.join(x)
    s = ""
    for i in xrange(len(kqcc)):
        try:
            x = kqcc[i+1]
        except IndexError:
            x = 'null'

        if kqcc[i][1] == 'B-PER\n':
            s = s + '  <PER>' + kqcc[i][0][0] + '</PER>  '
        elif kqcc[i][1] == 'B-LOC\n':
            if  x[1] == 'I-LOC\n':
                s = s + '  <LOC>' + kqcc[i][0][0]
            else:
                s = s + '  <LOC>'+kqcc[i][0][0]+'</LOC>  '
        elif kqcc[i][1] == 'I-LOC\n':
            if x[1] == 'I-LOC\n':
                s = s +" "+ kqcc[i][0][0]
            else:
                s = s + " " + kqcc[i][0][0] + '</LOC>  '
        elif kqcc[i][1] == 'B-ORG\n':
            if x[1] == 'I-ORG\n':
                s = s + '  <ORG>' + kqcc[i][0][0]
            else:
                s = s + '  <ORG>'+kqcc[i][0][0]+'</ORG>  '
        elif kqcc[i][1] == 'I-ORG\n':
            if x[1] == 'I-ORG\n':
                s = s + " " + kqcc[i][0][0]
            else:
                s = s + " " + kqcc[i][0][0] + '</ORG>  '
        else:
            s = s + " " +kqcc[i][0][0]
    return s + '\n' + s2


#crf add r0,r1,r-1
print "fitting model....."
crf = fit('crf_.pkl')
# print("Top positive:")
# state = crf.state_features_
# couter = Counter(state)
# most_cmn = couter.most_common(30)
# print_state_features(most_cmn)
#
# print("\nTop negative:")
# print_state_features(Counter(crf.state_features_).most_common()[-30:])


#optimize('crf.pkl')
print "estimating....."
estimate('crf_.pkl')


