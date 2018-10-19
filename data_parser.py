# -*- coding: utf-8 -*-

import logging
import pickle
import random
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from kmeans_transform import kmeans_transform

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
    vectorizer = TfidfVectorizer()
    feature = vectorizer.fit_transform(corpus)
    print(feature.shape)

    #k_means clustering
    num_clusters = 4
    feature = kmeans_transform(feature,num_clusters) 
    print(feature.shape)

    #make_training_set(author_pair, train_test_year)
