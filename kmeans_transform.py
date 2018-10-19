# -*- coding: utf-8 -*-

import logging
import numpy as np

from k_means import KMeans

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
logger = logging.getLogger("kmeans_transform")



def kmeans_transform(feature, num_clusters=100):
    feature = feature.T
    feature = feature.todense()
    feature = feature.tolist()
    logger.info("feature to list finished")
    k_means = KMeans(num_clusters, feature)
    k_means.main_loop()
    logger.info("k_means finished")
    clusters2matrix = np.mat(k_means.clusters[0])
    transformed_feat = clusters2matrix.sum(axis=0)
    for i in range(1,num_clusters):
        clusters2matrix = np.mat(k_means.clusters[i])
        new_feature = clusters2matrix.sum(axis=0)
        transformed_feat = np.row_stack((transformed_feat, new_feature))
    return transformed_feat.T





