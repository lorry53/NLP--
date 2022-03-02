import preprocessing as prepro
from textrank4zh import TextRank4Sentence
import jieba
import pandas as pd
import jieba.analyse
import wordcloud
import matplotlib.pyplot as plt


class keyinfo(object):

    def __init__(self):

        self.trks = TextRank4Sentence()
        self.prepro = prepro.preprocess()

        # 获取战略客户列表
        self.client_name = pd.read_csv("data_source/客户全称&简称.csv")
        self.name_dict = dict(zip(self.client_name["全称"].tolist(), self.client_name["简称"].tolist()))

        # jieba选择停用词库
        jieba.analyse.set_stop_words("data_source/停用词.txt")

    # jieba提取关键词，tfidf算法/textrank算法

    def jieba_keyword(self, tokenized_text, keyword_num, algo="tfidf"):

        corpus = " ".join(tokenized_text)

        if algo == "tfidf":
            keywords = jieba.analyse.extract_tags(corpus, topK=keyword_num, withWeight=False,
                                                  allowPOS=("ns", "n", "vn", "v", "nr"))
        if algo == "textrank":
            keywords = jieba.analyse.textrank(corpus, topK=keyword_num, withWeight=False,
                                              allowPOS=("ns", "n", "vn", "v", "nr"))

        # 如果提取的关键词中不包含公司名称，加入公司的全称
        for i in range(len(self.client_name)):
            if self.client_name["全称"][i] in tokenized_text and self.client_name["全称"][i] not in keywords:
                keywords.append(self.client_name["全称"][i])
            if str(self.client_name["简称"][i]) != "nan":
                continue
            elif self.client_name["简称"][i] in tokenized_text and self.client_name["简称"][i] not in keywords and \
                    self.client_name["全称"][i] not in keywords:
                keywords.append(self.client_name["简称"][i])

        return " ".join(keywords)


    # TextRank提取关键句
    def textrank_keysentence(self, corpus, keysentence_num):
        self.trks.analyze(corpus, lower=True, source='all_filters')
        result = self.trks.get_key_sentences(num=keysentence_num, sentence_min_len=6)
        keysentence = [i["sentence"] for i in result]

        return "\n".join(keysentence)

# 词频统计功能
class word_count(object):

    def __init__(self):
        pass

    def count_word(self,tokenized_text):
        counts = {}
        for word in tokenized_text:
            counts[word] = counts.get(word, 0) + 1
        counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)

        return counts

    def wordcloud(self, tokenized_text):
        corpus = " ".join(tokenized_text)

        wcloud = wordcloud.WordCloud(
            background_color='white',  # 背景颜色，根据图片背景设置，默认为黑色
            # mask = backgroup_Image, #笼罩图
            font_path='C:\Windows\Fonts\STZHONGS.TTF',  # 若有中文需要设置才会显示中文
            width=1000,
            height=860,
            margin=2).generate(corpus)  # generate 可以对全部文本进行自动分词
        # 参数 width，height，margin分别对应宽度像素，长度像素，边缘空白处

        plt.imshow(wcloud)
        plt.axis('off')
        plt.show()

        # 保存图片：默认为此代码保存的路径
        wordcloud.WordCloud.to_file(filename='词云.jpg')