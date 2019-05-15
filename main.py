from pyvi import ViTokenizer #For split vietnamese words
import pandas as pd #For reading xlsx file
from gensim.parsing.preprocessing import strip_non_alphanum, strip_multiple_whitespaces,preprocess_string, split_alphanum, strip_short, strip_numeric
import re #For preprocessing raw text
from sklearn.metrics import accuracy_score
import numpy as np
import xlrd

def compare(predict_spam, predict_non_spam, log):
    while (log[0] > log[1]):
        predict_spam /= 10
        log[0] -= 1
        if predict_spam > predict_non_spam:
            return True

    while (log[1] > log[0]):
        predict_non_spam /= 10
        log[1] -= 1
        if predict_non_spam > predict_spam:
            return False

    if predict_spam > predict_non_spam:
        return True
    return False


def predict(mail):
    mail = raw_text_preprocess(mail)

    vector = np.zeros(len(set_words))
    for i, word in enumerate(set_words):
        if word in mail:
            vector[i] = 1
    log = np.zeros(2)

    predict_spam = spam_coef
    predict_non_spam = non_spam_coef

    for i, v in enumerate(vector):
        if v == 0:
            predict_spam *= bayes_matrix[i][2]
            predict_non_spam *= bayes_matrix[i][3]
        else:
            predict_spam *= bayes_matrix[i][0]
            predict_non_spam *= bayes_matrix[i][1]

        if predict_spam < 1e-10:
            predict_spam *= 1000
            log[0] += 1

        if predict_non_spam < 1e-10:
            predict_non_spam *= 1000
            log[1] += 1

    if compare(predict_spam, predict_non_spam, log):
        return 1
    return 0

def smoothing(a, b):
    return float((a+1)/(b+1))

def raw_text_preprocess(raw):
    raw = re.sub(r"http\S+", "", raw)
    raw = strip_non_alphanum(raw).lower().strip()
    raw = split_alphanum(raw)
    raw = strip_short(raw, minsize=2)
    raw = strip_numeric(raw)
    raw = ViTokenizer.tokenize(raw)
    return raw

file_location = "./data.xlsx"
xl_file = xlrd.open_workbook(file_location)
dfs = xl_file.sheet_by_index(0)

document = []
label = []

for rows in range(dfs.nrows):
    document.append( dfs.cell_value(rows, 0) )

for rows in range(dfs.nrows):
    label.append(dfs.cell_value(rows, 1))

document = [raw_text_preprocess(d) for d in document]

document_test = document[54:]
label_test = label[54:]
document = document[:33] + document[33:54]
label = label[:33] + label[33:54]

set_words = []

for doc in document:
    words = doc.split(' ')
    set_words += words
    set(set_words)
print(len(set_words))

vectors = []

for doc in document:
    vector = np.zeros(len(set_words))
    for i, word in enumerate(set_words):
        if word in doc:
            vector[i] = 1
    vectors.append(vector)
print(np.shape(vectors))

spam = 0
non_spam = 0
for l in label:
    if l == 1:
        spam += 1
    else:
        non_spam += 1
print(spam, non_spam)
spam_coef = smoothing(spam, (spam+non_spam))
non_spam_coef = smoothing(non_spam, (spam+non_spam))


bayes_matrix = np.zeros((len(set_words), 4)) #app/spam, app/nonspam, nonapp/spam, nonapp/nonspam
for i, word in enumerate(set_words):
    app_spam = 0
    app_nonspam = 0
    nonapp_spam = 0
    nonapp_nonspam = 0
    for k, v in enumerate(vectors):
        if v[i] == 1:
            if label[k] == 1:
                app_spam += 1
            else:
                app_nonspam += 1
        else:
            if label[k] == 1:
                nonapp_spam += 1
            else:
                nonapp_nonspam += 1

    bayes_matrix[i][0] = smoothing(app_spam, spam)
    bayes_matrix[i][1] = smoothing(app_nonspam, non_spam)
    bayes_matrix[i][2] = smoothing(nonapp_spam, spam)
    bayes_matrix[i][3] = smoothing(nonapp_nonspam, non_spam)


for d in document_test:
    if( predict(d)==1):
        print("spam")
    else:
        print("non-spam")



