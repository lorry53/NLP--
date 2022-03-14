from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import numpy as np


class topic_cluster(object):

    def __init__(self, df):
        self.df = df
        self.df_final = self.df.reset_index().iloc[0:1][["corp_abbr", "time"]]


    '''vectorize the input documents'''

    def tfidf_vector(self):
        # 利用train-corpus提取特征
        corpus_train = self.df["content"].tolist()
        count_v1 = CountVectorizer()
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
        # self.df["clusters"] = km.labels_.tolist()
        order_centroids = km.cluster_centers_.argsort()[:, ::-1]


        cluster = 1
        for ind in order_centroids:  # 每个聚类选10个词
            words = []
            for index in ind[:10]:
                words.append(word_dict[index])
            self.df_final["topic" + str(cluster)] = " ".join(words)
            # self.df["topic" + str(cluster)] = ""
            # self.df["topic" + str(cluster)].iloc[0] = " ".join(words)
            cluster += 1

        return self.df_final

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