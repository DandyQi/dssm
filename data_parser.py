# -*- coding: utf-8 -*-

import logging
import pickle
import random
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import string
from sklearn.metrics.pairwise import pairwise_distances
import math
from nltk.util import ngrams
from itertools import permutations,groupby

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
logger = logging.getLogger("data_parser")


def make_paper_mapping(data):
    author_list = []
    paper_list = []
    year_list = []
    for idx, item in data.iterrows():
        authors = item["author"].split(",")
        author_list.extend(authors)
        paper_list.extend([idx] * len(authors))
        year_list.extend([item["year"]] * len(authors))

    return pd.DataFrame(np.array([author_list, paper_list, year_list]).T, columns=["author", "paper_idx", "year"])


def make_domain_mapping(data):
    author_list = []
    domain_list = []
    for idx, item in data.iterrows():
        authors = item["author"].split(",")
        author_list.extend(authors)
        domain_list.extend([item["domain"]] * len(authors))

    return pd.DataFrame(np.array([author_list, domain_list]).T, columns=["author", "domain"]).drop_duplicates()


def make_author_pair(sub, obj):
    sub_author_list = domain2author[domain2author["domain"] == sub]
    obj_author_list = domain2author[domain2author["domain"] == obj]

    common_author_list = pd.merge(left=sub_author_list, right=obj_author_list, how="inner", on="author")["author"]
    logger.info("the size of collaboration author: %s" % len(common_author_list))
    
    pair = []
    for author in common_author_list:
        paper_list = author2paper[author2paper["author"] == author]["paper_idx"].values.astype(np.int32)
        paper = data_set.iloc[paper_list, :]
        collaboration_author = paper[paper["domain"] == obj]
        for idx, item in collaboration_author.iterrows():
            for au in item["author"].split(","):
                if au != author:
                    pair.append([author, au, item["year"]])
    pair = pd.DataFrame(np.array(pair), columns=["sub_author", "obj_author", "year"])
    pair["year"] = pair["year"].astype(np.int32)
    return pair


def make_feature(author):
    papers = author2paper[author2paper["author"] == author]
    papers["year"] = papers["year"].astype(np.int32)
    paper_list = papers[papers["year"] <= train_test_year]["paper_idx"].values.astype(np.int32)
    feat = np.mean(feature[paper_list].toarray(), axis=0)
    return feat


def save_data(pair, name):
    query_list = []
    pos_list = []
    for idx, item in pair.iterrows():
        query_list.append(make_feature(item["sub_author"]))
        pos_list.append(make_feature(item["obj_author"]))

    query = np.array(query_list)
    pos = np.array(pos_list)

    pickle.dump(query, open("data/%s_query" % name, "wb"), protocol=2)
    pickle.dump(pos, open("data/%s_positive" % name, "wb"), protocol=2)

    neg = feature[np.array(random.sample(range(data_set.shape[0]), pair.shape[0]))].toarray()
    pickle.dump(neg, open("data/%s_negative" % name, "wb"), protocol=2)


def make_training_set(pair, year):
    training_pair = pair[pair["year"] <= year]
    temp_pair = pair[pair["year"] > year]

    eval_pair = []
    author = training_pair["sub_author"].drop_duplicates().values
    for au in author:
        if temp_pair[temp_pair["sub_author"] == au].shape[0] >= 5:
            eval_pair.append(temp_pair[temp_pair["sub_author"] == au])
    eval_pair = pd.concat(eval_pair, ignore_index=True)
    eval_pair = pd.DataFrame(eval_pair)

    logger.info("the size of training pair: %s" % training_pair.shape[0])
    logger.info("the size of eval pair: %s" % eval_pair.shape[0])

    save_data(training_pair, "train")
    save_data(eval_pair, "eval")

    logger.info("save data")


def year_str2int(row):
    row["year"] = int(row["year"])
    return row


def concat(row):
    row["content"] = "%s %s" % (row["title"], row["abstract"])
    return row


def word_count(file_list):
    import collections
    word_freq = collections.defaultdict(int)
    for file in file_list:
        with open(file) as f:
            for l in f:
                for w in l.strip().split():  
                    word_freq[w] += 1
    return word_freq

def build_dict(file_name, min_word_freq=50):
    word_freq = word_count(file_name) 
   # word_freq = filter(lambda x: x[1] > min_word_freq, word_freq.items()) # filter将词频数量低于指定值的单词删除。
    word_freq_sorted = sorted(word_freq, key=lambda x: (-x[1], x[0]))
    # key用于指定排序的元素，因为sorted默认使用list中每个item的第一个元素从小到
    #大排列，所以这里通过lambda进行前后元素调序，并对词频去相反数，从而将词频最大的排列在最前面
    words, _ = list(zip(*word_freq_sorted))
    word_idx = dict(zip(words, xrange(len(words))))
    word_idx['<unk>'] = len(words) #unk表示unknown，未知单词
    return word_idx

# def deleteDuplicatedElementFromList3(listA):
#     # return list(set(listA))
#     return sorted(set(listA), key=listA.index)

def deleteDuplicatedElementFromList3(l3g):
    new_list = []
    l3g.sort()
    l3g = groupby(l3g)
    for k, m in l3g:
        new_list.extend(k)
    return sorted(set(new_list), key=new_list.index)
def word_to_ngrams(word,ngram_size=3,lower=True):
    """Returns a list of all n-gram possibilities of the given word."""
    if lower:
        word = word.lower()
    word = '<' + word + '>'
    return list(map(lambda x: ''.join(x), list(ngrams(word,ngram_size))))

def letter3gramtable(corpus):
    l3g = []
    for line in corpus:
        remove_punct_map = dict.fromkeys(map(ord, string.punctuation))
        #去除标点符号
        line = line.translate(remove_punct_map)
        for word in line.split():
            #print(word)
            w2n = word_to_ngrams(word)
            l3g.extend(w2n)
    print('Before remove duplication,length:',len(l3g))
    l3g = deleteDuplicatedElementFromList3(l3g)
    print('Remove duplication,length:',len(l3g))
    return l3g

# def letter3gramtable():
#     vocabulary = ['a','b','c','d','e','f','g','h','i','j','k','l','m',\
#                     'n','o','p','q','r','s','t','u','v','w','x','y','z']
#     w2n_3 = list(permutations(vocabulary,3))
#     l3g = []
#     for i in w2n_3:
#         temp = "".join(list(i))
#         l3g.append(temp)
#     w2n_2 = list(permutations(vocabulary,2))
#     l2g = []
#     for i in w2n_2:
#         temp = "".join(list(i))
#         l2g.append(temp)
#     l2g_left =[]
#     l2g_right = []
#     for i in l2g:
#         l2g_left.append('<' + i) 
#     for i in l2g:
#         l2g_right.append(i + '>')
#     l3g.extend(l2g_left)
#     l3g.extend(l2g_right)
#     return l3g

def letter3grams(corpus):
    l3g = letter3gramtable(corpus)
    feature = np.zeros((len(corpus),len(l3g)))
    i = 0
    for line in corpus:
        if i % 100 == 0:
            print('Transformed lines:%d' %i)
        remove_punct_map = dict.fromkeys(map(ord, string.punctuation))
        #去除标点符号
        line = line.translate(remove_punct_map)
        line_l3g_count = np.zeros(len(l3g), dtype='int')
        for word in line.split():
            word_l3g_count = np.zeros(len(l3g), dtype='int')
            w3g = word_to_ngrams(word)
            for j in w3g:
                # if j in l3g:
                    idx = l3g.index(j)
                    word_l3g_count[idx] += 1

            # print (V1)
            line_l3g_count += word_l3g_count
        
        feature[i] = line_l3g_count
        i += 1
    return feature

def lsh_transform(feature,k = 2000):
    # feature = feature.todense()

    trans_mat = np.random.randn(feature.shape[1],k)
    hash_mat = np.dot(feature,trans_mat)
    hash_mat = np.array(hash_mat)
    for i in range(len(hash_mat)):
        for j in range(k):
            if hash_mat[i][j] >= 0. :
                hash_mat[i][j] = 1
            else:
                hash_mat[i][j] = 0
    return hash_mat

def lsh_cosine(feature):
    hash_mat = lsh_transform(feature)
    hamming_distance = len(np.nonzero(hash_mat[17]-hash_mat[19])[0])
    cos_distance = math.cos(hamming_distance/len(hash_mat[0])*math.pi)
    return cos_distance

if __name__ == "__main__":
    file_list = [
        "Cross-Domain_data/Data Mining.txt",
        "Cross-Domain_data/Database.txt",
        "Cross-Domain_data/Medical Informatics.txt",
        "Cross-Domain_data/Theory.txt",
        "Cross-Domain_data/Visualization.txt"
    ]

    train_test_year = 2001

    df_list = []
    for file in file_list:
        df = pd.read_csv(file, sep="\t", names=["conference/journal", "title", "author", "year", "abstract"]).dropna()
        df["domain"] = file.split("/")[1].split(".")[0]
        df_list.append(df)
    data_set = pd.concat(df_list, ignore_index=True)
    data_set = pd.DataFrame(data_set)

    data_set = data_set.apply(year_str2int, axis=1)
    data_set = data_set.apply(concat, axis=1)

    logger.info("the size of data set: %s" % data_set.shape[0])
    author2paper = make_paper_mapping(data_set)
    domain2author = make_domain_mapping(data_set)

    sub_view = "Data Mining"
    obj_view = "Theory"

    author_pair = make_author_pair(sub_view, obj_view)

    corpus = data_set["content"].values
    # print(corpus.split(","))
    # vectorizer = TfidfVectorizer()
    # feature = vectorizer.fit_transform(corpus)
    # print(feature.shape)
    # make_training_set(author_pair, train_test_year)
    # with open('./Cross-Domain_data/corpus.txt','a') as f:
    #     for content in corpus:
    #         f.write(str(content)+'\r\n')

    feature = letter3grams(corpus)
    print(feature.shape)
    print(feature[0])
    
    # print(1-pairwise_distances(feature[0:20],metric="cosine")[17][19])
    # print(lsh_cosine(feature))
    # make_training_set(author_pair, train_test_year)