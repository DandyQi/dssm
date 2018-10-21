# -*- coding: utf-8 -*-

import logging
import pickle
import random
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from l3wtransformer import L3wTransformer
import string
import codecs

l3wt = L3wTransformer()

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
logger = logging.getLogger("data_parser")


#进行作者与论文的匹配，即拆分作者列表，得到该作者都发表过哪些论文（基于现有数据无法判断是否重名）,字段为[作者名，论文编号，年份]
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


#进行作者与领域的匹配，得到一个领域下有哪些作者，字段为[作者名，领域]
def make_domain_mapping(data):
    author_list = []
    domain_list = []
    for idx, item in data.iterrows():
        authors = item["author"].split(",")
        author_list.extend(authors)
        domain_list.extend([item["domain"]] * len(authors))

    return pd.DataFrame(np.array([author_list, domain_list]).T, columns=["author", "domain"]).drop_duplicates()


#暂定主领域为Data Mining，客领域为Theory，merge两个领域的作者列表，得到有跨域合作的作者对，字段为[作者，合作作者，年份]
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


#构造并保存训练集与测试集，采用pickle的方式以二进制打包数据，分为训练时的query表、正例表、负例表，测试同理；此时的数据表为一个二维numpy array，维度为[作者对数，特征维数]
def make_feature(author):
    papers = author2paper[author2paper["author"] == author]
    papers["year"] = papers["year"].astype(np.int32)
    paper_list = papers[papers["year"] <= train_test_year]["paper_idx"].values.astype(np.int32)
    feat = np.mean(feature[paper_list], axis=0)
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

    neg = feature[np.array(random.sample(range(data_set.shape[0]), pair.shape[0]))]
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

def deleteDuplicatedElementFromList3(listA):
    # return list(set(listA))
    return sorted(set(listA), key=listA.index)

def letter3garmtable(corpus):
    A = []
    for line in open('C:/biancheng/DSSM/dssm-master/corpus.txt', 'r', encoding='UTF-8').readlines():
        remove_punct_map = dict.fromkeys(map(ord, string.punctuation))
        corpus1 = line.translate(remove_punct_map)
        for word in corpus1.split():
            #print(word)
            B = l3wt.word_to_ngrams(word)
            A[len(A):len(A)] = B
    letter3garmtable = deleteDuplicatedElementFromList3(A)
    print(len(letter3garmtable))
    return letter3garmtable

def letter3grams(corpus):
    C = letter3garmtable(corpus)
    feature = np.zeros((len(corpus),len(C)))
    i = 0
    for line in open('C:/biancheng/DSSM/dssm-master/corpus.txt', 'r', encoding='UTF-8').readlines():
        remove_punct_map = dict.fromkeys(map(ord, string.punctuation))
        corpus1 = line.translate(remove_punct_map)
        V2 = np.zeros(len(C), dtype='int')
        for word in corpus1.split():
            V1 = np.zeros(len(C), dtype='int')
            D = l3wt.word_to_ngrams(word)
            for j in range(len(D)):
                if D[j] in C:
                    n = C.index(D[j])
                    V1[n] = V1[n] + 1
            # print (V1)
            V2 = V2 + V1
        for m in range(len(C)):
            feature[i,m] = V2[m]
        i = i + 1
    return feature




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

#采用letter trigrams抽取文本特征
    corpus = data_set["content"]
    # print(len(corpus2))
    # resName = "C:/biancheng/DSSM/dssm-master/corpus.txt"
    # result = codecs.open(resName, 'w', 'utf-8')
    # for i in range(len(corpus2)):
    #     result.write(corpus2[i]+'\r\n')
    # vectorizer = TfidfVectorizer()
    # feature = vectorizer.fit_transform(corpus)
    feature = letter3grams(corpus)
    print(feature.shape)
    print(feature)
    make_training_set(author_pair, train_test_year)
