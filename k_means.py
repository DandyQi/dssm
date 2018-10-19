# -*- coding: utf-8 -*-

import logging
import random

from similarity import similarity

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
logger = logging.getLogger("k_means")


class KMeans(object):
    """K-Means clustering. Uses cosine similarity as the distance function."""

    def __init__(self, k, vectors):
        assert len(vectors) >= k
        self.centers = random.sample(vectors, k)
        self.clusters = [[] for c in self.centers]
        self.vectors = vectors

    def update_clusters(self):
        """Determine which cluster center each `self.vector` is closest to."""
        def closest_center_index(vector):
            """Get the index of the closest cluster center to `self.vector`."""
            similarity_to_vector = lambda center: similarity(center,vector)
            center = max(self.centers, key=similarity_to_vector)
            return self.centers.index(center)

        self.clusters = [[] for c in self.centers]
        for vector in self.vectors:
             index = closest_center_index(vector)
             self.clusters[index].append(vector)

    def update_centers(self):
        """Move `self.centers` to the centers of `self.clusters`.

        Return True if centers moved, else False.

        """
        new_centers = []
        for cluster in self.clusters:
            center = [average(ci) for ci in zip(*cluster)]
            new_centers.append(center)

        if new_centers == self.centers:
            return False

        self.centers = new_centers
        return True

    def main_loop(self):
        """Perform k-means clustering."""
        logger.info("k_means begin")
        self.update_clusters()
        num = 1
        logger.info("update %s" % str(num))
        while self.update_centers():
            self.update_clusters()
            num += 1
            logger.info("update %s" % str(num))


def average(sequence):
    return sum(sequence) / len(sequence)
