from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import numpy as np


class topic_cluster(object):

    def __init__(self, df):
        self.df = df

    '''vectorize the input documents'''

    def tfidf_vector(self):
        # 利用train-corpus提取特征
        corpus_train = self.df["content"].tolist()
        count_v1 = CountVectorizer(max_df=0.4, min_df=0.01)
        counts_train = count_v1.fit_transform(corpus_train)

        word_dict = {}
        for index, word in enumerate(count_v1.get_feature_names()):
            word_dict[index] = word

        tfidftransformer = TfidfTransformer()
        tfidf_train = tfidftransformer.fit(counts_train).transform(counts_train)
        return tfidf_train, word_dict

    '''topic cluster'''

    def cluster_kmeans(self, tfidf_train, word_dict, num_clusters = 7):  # K均值分类
        km = KMeans(n_clusters=num_clusters)
        km.fit(tfidf_train)
        self.df["clusters"] = km.labels_.tolist()
        order_centroids = km.cluster_centers_.argsort()[:, ::-1]

        f_clusterwords = open("data_output/cluster_keywords.txt", 'w+')
        cluster = 1
        for ind in order_centroids:  # 每个聚类选 50 个词
            words = []
            for index in ind[:50]:
                words.append(word_dict[index])
            f_clusterwords.write(str(cluster) + '\t' + ','.join(words) + '\n')
            cluster += 1
        f_clusterwords.close()

        return self.df

    '''select the best cluster num'''

    def best_kmeans(self, tfidf_matrix):
        K = range(1, 20)
        meandistortions = []
        for k in K:
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(tfidf_matrix)
            meandistortions.append(
                sum(np.min(cdist(tfidf_matrix.toarray(), kmeans.cluster_centers_, 'euclidean'), axis=1)) /
                tfidf_matrix.shape[0])
        plt.plot(K, meandistortions, 'bx-')
        plt.grid(True)
        plt.xlabel('Number of clusters')
        plt.ylabel('Average within-cluster sum of squares')
        plt.title('Elbow for Kmeans clustering')
        plt.show()
        plt.savefig("data_output/best_kmeans.jpg")