import torch
from torch import nn
from torch.nn import functional as F
import pandas as pd
from collections import Counter
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from collections import Counter, defaultdict
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import re
from sklearn.utils import shuffle



def cos_dist(x, y):
    ## cosine distance function
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    batch_size = x.size(0)
    c = torch.clamp(1 - cos(x.view(batch_size, -1), y.view(batch_size, -1)),
                    min=0)
    return c.mean()




def tag_mapping(tags):
    """
    Create a dictionary and a mapping of tags, sorted by frequency.
    """
    #tags = [s[1] for s in dataset]
    dico = Counter(tags)
    tag_to_id, id_to_tag = create_mapping(dico)
    print("Found %i unique named entity tags" % len(dico))
    return dico, tag_to_id, id_to_tag


def create_mapping(dico):
    """
    Create a mapping (item to ID / ID to item) from a dictionary.
    Items are ordered by decreasing frequency.
    """
    sorted_items = sorted(dico.items(), key=lambda x: (-x[1], x[0]))
    id_to_item = {i: v[0] for i, v in enumerate(sorted_items)}
    item_to_id = {v: k for k, v in id_to_item.items()}
    return item_to_id, id_to_item




def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def clean_doc(x, word_freq):
    stop_words = set(stopwords.words('english'))
    clean_docs = []
    most_commons = dict(word_freq.most_common(min(len(word_freq), 50000)))
    for doc_content in x:
        doc_words = []
        cleaned = clean_str(doc_content.strip())
        for word in cleaned.split():
            if word not in stop_words and word_freq[word] >= 5:
                if word in most_commons:
                    doc_words.append(word)
                else:
                    doc_words.append("<UNK>")
        doc_str = ' '.join(doc_words).strip()
        clean_docs.append(doc_str)
    return clean_docs



def load_dataset(dataset):

    if dataset == 'sst':
        df_train = pd.read_csv("./dataset/sst/SST-2/train.tsv", delimiter='\t', header=0)
        
        df_val = pd.read_csv("./dataset/sst/SST-2/dev.tsv", delimiter='\t', header=0)
        
        df_test = pd.read_csv("./dataset/sst/SST-2/sst-test.tsv", delimiter='\t', header=None, names=['sentence', 'label'])

        train_sentences = df_train.sentence.values
        val_sentences = df_val.sentence.values
        test_sentences = df_test.sentence.values
        train_labels = df_train.label.values
        val_labels = df_val.label.values
        test_labels = df_test.label.values   
    

    if dataset == '20news':
        
        VALIDATION_SPLIT = 0.8
        newsgroups_train  = fetch_20newsgroups('dataset/20news', subset='train',  shuffle=True, random_state=0)
        print(newsgroups_train.target_names)
        print(len(newsgroups_train.data))

        newsgroups_test  = fetch_20newsgroups('dataset/20news', subset='test',  shuffle=False)

        print(len(newsgroups_test.data))

        train_len = int(VALIDATION_SPLIT * len(newsgroups_train.data))

        train_sentences = newsgroups_train.data[:train_len]
        val_sentences = newsgroups_train.data[train_len:]
        test_sentences = newsgroups_test.data
        train_labels = newsgroups_train.target[:train_len]
        val_labels = newsgroups_train.target[train_len:]
        test_labels = newsgroups_test.target



    if dataset == '20news-15':
        VALIDATION_SPLIT = 0.8
        cats = ['alt.atheism',
        'comp.graphics',
        'comp.os.ms-windows.misc',
        'comp.sys.ibm.pc.hardware',
        'comp.sys.mac.hardware',
        'comp.windows.x',
        'rec.autos',
        'rec.motorcycles',
        'rec.sport.baseball',
        'rec.sport.hockey',
        'misc.forsale',
        'sci.crypt',
        'sci.electronics',
        'sci.med',
        'sci.space']
        newsgroups_train  = fetch_20newsgroups('dataset/20news', subset='train',  shuffle=True, categories=cats, random_state=0)
        print(newsgroups_train.target_names)
        print(len(newsgroups_train.data))

        newsgroups_test  = fetch_20newsgroups('dataset/20news', subset='test',  shuffle=False, categories=cats)

        print(len(newsgroups_test.data))

        train_len = int(VALIDATION_SPLIT * len(newsgroups_train.data))

        train_sentences = newsgroups_train.data[:train_len]
        val_sentences = newsgroups_train.data[train_len:]
        test_sentences = newsgroups_test.data
        train_labels = newsgroups_train.target[:train_len]
        val_labels = newsgroups_train.target[train_len:]
        test_labels = newsgroups_test.target


    if dataset == '20news-5':
        cats = [
        'soc.religion.christian',
        'talk.politics.guns',
        'talk.politics.mideast',
        'talk.politics.misc',
        'talk.religion.misc']
              
        newsgroups_test  = fetch_20newsgroups('dataset/20news', subset='test',  shuffle=False, categories=cats)
        print(newsgroups_test.target_names)
        print(len(newsgroups_test.data))

        train_sentences = None
        val_sentences = None
        test_sentences = newsgroups_test.data
        train_labels = None
        val_labels = None
        test_labels = newsgroups_test.target

    if dataset == 'wos':
        TESTING_SPLIT = 0.6
        VALIDATION_SPLIT = 0.8
        file_path = './dataset/WebOfScience/WOS46985/X.txt'
        with open(file_path, 'r') as read_file:
            x_temp = read_file.readlines()
            x_all = []
            for x in x_temp:
                x_all.append(str(x))

        print(len(x_all))

        file_path = './dataset/WebOfScience/WOS46985/Y.txt'
        with open(file_path, 'r') as read_file:
            y_temp= read_file.readlines()
            y_all = []
            for y in y_temp:
                y_all.append(int(y))
        print(len(y_all))
        print(max(y_all), min(y_all))


        x_in = []
        y_in = []
        for i in range(len(x_all)):
            x_in.append(x_all[i])
            y_in.append(y_all[i])


        train_val_len = int(TESTING_SPLIT * len(x_in))
        train_len = int(VALIDATION_SPLIT * train_val_len)

        train_sentences = x_in[:train_len]
        val_sentences = x_in[train_len:train_val_len]
        test_sentences = x_in[train_val_len:]

        train_labels = y_in[:train_len]
        val_labels = y_in[train_len:train_val_len]
        test_labels = y_in[train_val_len:]

        print(len(train_labels))
        print(len(val_labels))
        print(len(test_labels))


    if dataset == 'wos-100':
        TESTING_SPLIT = 0.6
        VALIDATION_SPLIT = 0.8
        file_path = './dataset/WebOfScience/WOS46985/X.txt'
        with open(file_path, 'r') as read_file:
            x_temp = read_file.readlines()
            x_all = []
            for x in x_temp:
                x_all.append(str(x))

        print(len(x_all))

        file_path = './dataset/WebOfScience/WOS46985/Y.txt'
        with open(file_path, 'r') as read_file:
            y_temp= read_file.readlines()
            y_all = []
            for y in y_temp:
                y_all.append(int(y))
        print(len(y_all))
        print(max(y_all), min(y_all))


        x_in = []
        y_in = []
        for i in range(len(x_all)):
            if y_all[i] in range(100):
                x_in.append(x_all[i])
                y_in.append(y_all[i])

        for i in range(133):
            num = 0
            for y in y_in:
                if y == i:
                    num = num + 1
            # print(num)

        train_val_len = int(TESTING_SPLIT * len(x_in))
        train_len = int(VALIDATION_SPLIT * train_val_len)

        train_sentences = x_in[:train_len]
        val_sentences = x_in[train_len:train_val_len]
        test_sentences = x_in[train_val_len:]

        train_labels = y_in[:train_len]
        val_labels = y_in[train_len:train_val_len]
        test_labels = y_in[train_val_len:]

        print(len(train_labels))
        print(len(val_labels))
        print(len(test_labels))

    if dataset == 'wos-34':
        TESTING_SPLIT = 0.6
        VALIDATION_SPLIT = 0.8
        file_path = './dataset/WebOfScience/WOS46985/X.txt'
        with open(file_path, 'r') as read_file:
            x_temp = read_file.readlines()
            x_all = []
            for x in x_temp:
                x_all.append(str(x))

        print(len(x_all))

        file_path = './dataset/WebOfScience/WOS46985/Y.txt'
        with open(file_path, 'r') as read_file:
            y_temp= read_file.readlines()
            y_all = []
            for y in y_temp:
                y_all.append(int(y))
        print(len(y_all))
        print(max(y_all), min(y_all))

        x_in = []
        y_in = []
        for i in range(len(x_all)):
            if (y_all[i] in range(100)) != True:
                x_in.append(x_all[i])
                y_in.append(y_all[i])

        for i in range(133):
            num = 0
            for y in y_in:
                if y == i:
                    num = num + 1
            # print(num)

        train_val_len = int(TESTING_SPLIT * len(x_in))
        train_len = int(VALIDATION_SPLIT * train_val_len)

        train_sentences = None
        val_sentences = None
        test_sentences = x_in[train_val_len:]
        
        train_labels = None
        val_labels = None
        test_labels = y_in[train_val_len:]

        print(len(test_labels))
        
    if dataset == 'agnews':

        VALIDATION_SPLIT = 0.8
        labels_in_domain = [1, 2]

        train_df = pd.read_csv('./dataset/agnews/train.csv', header=None)
        train_df.rename(columns={0: 'label',1: 'title', 2:'sentence'}, inplace=True)
        # train_df = pd.concat([train_df, pd.get_dummies(train_df['label'],prefix='label')], axis=1)
        print(train_df.dtypes)
        train_in_df_sentence = []
        train_in_df_label = []
        
        for i in range(len(train_df.sentence.values)):
            sentence_temp = ''.join(str(train_df.sentence.values[i]))
            train_in_df_sentence.append(sentence_temp)
            train_in_df_label.append(train_df.label.values[i]-1)

        test_df = pd.read_csv('./dataset/agnews/test.csv', header=None)
        test_df.rename(columns={0: 'label',1: 'title', 2:'sentence'}, inplace=True)
        # test_df = pd.concat([test_df, pd.get_dummies(test_df['label'],prefix='label')], axis=1)
        test_in_df_sentence = []
        test_in_df_label = []
        for i in range(len(test_df.sentence.values)):
            test_in_df_sentence.append(str(test_df.sentence.values[i]))
            test_in_df_label.append(test_df.label.values[i]-1)

        train_len = int(VALIDATION_SPLIT * len(train_in_df_sentence))

        train_sentences = train_in_df_sentence[:train_len]
        val_sentences = train_in_df_sentence[train_len:]
        test_sentences = test_in_df_sentence
        train_labels = train_in_df_label[:train_len]
        val_labels = train_in_df_label[train_len:]
        test_labels = test_in_df_label
        print(len(train_sentences))
        print(len(val_sentences))
        print(len(test_sentences))


    return train_sentences, val_sentences, test_sentences, train_labels, val_labels, test_labels

        