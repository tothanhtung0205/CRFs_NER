# coding=utf-8
from io import open
import sklearn_crfsuite
from sklearn_crfsuite import metrics
from sklearn.metrics import make_scorer
import scipy.stats
from pyvi.pyvi import ViPosTagger,ViTokenizer
from sklearn.externals import joblib
from sklearn.model_selection import RandomizedSearchCV

#Featurelize sentences
def contains_digit(str):
    for char in str:
        if char.isdigit():
            return True
    return False

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

def single_features(sent, i):
    word_0 = sent[i][0].lower()
    postag_0 = sent[i][1]

    word_minus_1 = sent[i-1][0].lower() if i>0 else "BOS"
    postag_minus_1 = sent[i-1][1] if i>0 else "BOS"

    word_minus_2 = sent[i-2][0].lower() if i>1 else "BOS"
    postag_minus_2 = sent[i-2][1] if i>1 else "BOS"

    word_add_1 = sent[i+1][0].lower() if i<len(sent)-1 else "EOS"
    postag_add_1 = sent[i+1][1] if i < len(sent)-1 else "EOS"

    word_add_2 = sent[i + 2][0] if i < len(sent)-2  else "EOS"
    postag_add_2 = sent[i + 2][1] if i < len(sent)-2 else "EOS"

    O_0 = {
        'W(0).isupper()': word_0.isupper(),  # O_0
        'W(0).istitle()': word_0.istitle(),  # 0_0
        'W(0).isdigit()': word_0.isdigit(),  # 0_0
        'W(0).contains_digit()': contains_digit(word_0),  # O_0
        'W(0).is_full_name()':is_full_name(word_0)
        }

    #W0_O0 = O_0
    #W0_O0.update({'W(0)': word_0})

    features = {
        'bias': 1.0,
        'W(0)': word_0,  # W_0,
        'P(0)': postag_0,  # P_0
        'O(0)': O_0,

        'W(-1)':word_minus_1,
        'P(-1)':postag_minus_1,

        'W(-2)':word_minus_2,
        'P(-2)':postag_minus_2,

        'W(+1)':word_add_1,
        'P(+1)':postag_add_1,

        'W(+2)':word_add_2,
        'P(+2)':postag_add_2,

        'W(-1)+W(0)':word_minus_1+"+"+word_0,
        'W(0)+W(1)' :word_0+"+"+word_add_1,
        'W(-2)+W(-1)':word_minus_2+"+"+word_minus_1,
        'W(1)+W(2)':word_add_1+"+"+word_add_2,


        'P(-1)+P(0)':postag_minus_1+'+'+postag_0,
        'P(0)+P(1)':postag_0+'+' +postag_add_1,
        'P(-2)+P(-1)':postag_minus_2+'+'+postag_minus_1,
        'P(1)+P(2)':postag_add_1+'+'+postag_add_2,


        # 'W(-1)+W(0)+W(1)':word_minus_1+'+'+word_0+'+'+word_add_1,
        # 'W(-2)+W(-1)+W(0)':word_minus_2+'+'+word_minus_1+'+'+word_0,
        # 'W(0)+W(1)+W(2)':word_0+"+"+word_add_1+'+'+word_add_2,

        'W(0)+P(0)':word_0+'+'+postag_0,
        'W(0)+P(1)':word_0+'+'+postag_add_1,
        'W(0)+P(-1)': word_0 + '+' + postag_minus_1,

        #'W(0)+O(0)':W0_O0,
    }


    return features


def word2features(sent, i):
    features = {}
    features.update(single_features(sent,i))
    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [tup[2] for tup in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]

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
train_sents = read_file('dataset/train_nor.txt')

X_train = [sent2features(s) for s in train_sents]
y_train = [sent2labels(s) for s in train_sents]


def fit(model):
    #training phase
    try:
        crf = joblib.load(model)
        print 'load model completed !!!'
        return crf
    except: crf = None
    if crf == None:
        crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
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
    rs.fit(X_train, y_train)
    print('best params:', rs.best_params_)
    print('best CV score:', rs.best_score_)

def estimate(model):
    crf = joblib.load(model)
    test_sents = read_file('dataset/test_nor.txt')

    X_test = [sent2features(s) for s in test_sents]
    y_test = [sent2labels(s) for s in test_sents]

    labels = list(crf.classes_)
    labels.remove('O\n')
    y_pred = crf.predict(X_test)
    print y_pred
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
    # postaged_sent = ViPosTagger.postagging(ViTokenizer.tokenize(test_sent))
    postaged_sent = ViPosTagger.postagging(test_sent)

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
    #query = u"Cơ quan điều tra Công an quận Nam Từ Liêm (Hà Nội) xác định, Nguyễn Thị Thuận, Bí thư Đảng uỷ phường Mỹ Đình 1 cùng chồng là Nguyễn Văn Tiện tổ chức đánh bạc dưới hình thức ghi lô đề với số tiền lên đến hơn 4,7 tỉ đồng và thu lợi bất chính gần 600 triệu đồng"
    query = unicode(query, encoding='utf-8')
    kqcc = test_ner(crf, query)
    s = [x[0][0] + u' -- ' + unicode(x[1], 'utf-8') for x in kqcc]
    return u''.join(s)
fit('crf05.pkl')
optimize('crf05.pkl')

estimate('crf05.pkl')
