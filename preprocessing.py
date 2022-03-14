#!/usr/bin/env python
# coding: utf-8

from collections import Counter
import jieba
from bs4 import BeautifulSoup
import pandas as pd
import filter_sensitive_words as filter


class preprocess(object):

    def __init__(self):
        self.dfa = filter.DFAUtils()
        self.company_info = pd.read_csv("data_source/客户基本信息表.csv").drop(columns=["seq","created_at"])

        # 获取战略客户列表
        self.client_name = pd.read_csv("data_source/客户全称&简称.csv")
        self.name_dict = dict(zip(self.client_name["全称"].tolist(), self.client_name["简称"].tolist()))

        # 获取中文停用词表
        self.stop_words = []
        with open(r"data_source/停用词.txt", "r", encoding='UTF-8') as f:
            for line in f.readlines():
                self.stop_words.append(line.strip('\n'))

        # jieba中导入白名单和客户名称
        for word in self.name_dict.keys():
            jieba.add_word(word)
        for word in self.name_dict.values():
            if str(word) != "nan":
                jieba.add_word(word)

        # with open(r"data_source/白名单.txt", "r", encoding='UTF-8') as f:
        #     for line in f.readlines():
        #         jieba.add_word(line.strip('\n'))


    # 读入新闻数据
    def read_doc(self, doc_name):
        df = pd.read_csv(doc_name)
        df["content"] = df["1, 1000"].str.cat(
            [df["1001, 1000"], df["2001, 1000"], \
             df["3001, 1000"], df["4001, 1000"], \
             df["5001, 1000"], df["6001, 1000"], \
             df["7001, 1000"], df["8001, 1000"], \
             df["9001, 1000"]], sep=" ", na_rep=" ")
        df = df.drop(
            columns=["1, 1000", "1001, 1000", "2001, 1000", "3001, 1000",\
                     "4001, 1000", "5001, 1000", "6001, 1000", \
                     "7001, 1000", "8001, 1000", "9001, 1000"])
        return df


    # 解析html，doc_type代表文件类型
    def html_encoding(self, corpus, doc_type):
        # 创建BeautifulSoup对象
        soup = BeautifulSoup(corpus, 'lxml')
        clean_string = ""

        if doc_type == 0:
            for item in soup.find_all("p"):
                clean_string += item.get_text()

        if doc_type == 1:
            for item in soup.find_all("div"):
                clean_string += item.get_text()

        return clean_string


    # 标记需要删除的文本
    def del_check(self, corpus):

        delete = 0

        # DFA算法检测敏感词
        if self.dfa.is_contain(corpus):
            delete = 1

        # 去除长度过短的文本
        if len(corpus) < 20:
            delete = 1

        return delete


    # 文本清洗
    def clean_data(self, corpus):
        # 分词
        tokenized_text = jieba.lcut(corpus)
        # 去除停用词
        tokenized_text = [word for word in tokenized_text if word not in self.stop_words and word not in " "]
        # 去除低频词
        word_dict = Counter(tokenized_text)
        low_frequency_words = [[k for (k, v) in word_dict.items() if v < 2]]
        tokenized_text = [word for word in tokenized_text if word not in low_frequency_words]
        # 去除高频词
        high_frequency_words = [[k for (k, v) in word_dict.most_common(10)]]
        tokenized_text = [word for word in tokenized_text if word not in high_frequency_words]

        return tokenized_text

    # 筛选功能，type=0按战略客户名称筛选，type=1按日期筛选（需要提供start_time和end_time），type=2按行业筛选（提供行业清单）
    def filter_df(self, df: object, type: object, start_time: object = None, end_time: object = None, industry_list: object = None) -> object:

        if type == 0:
            df_new = df[df['corp_abbr'].isin(self.name_dict.keys())]
            return df_new

        if type == 1:
            df_new = df[(df['show_time'] >= start_time) & (df['show_time'] <= end_time)]
            return df_new

        if type == 2:
            df_new = df[df['corp_abbr'].isin(industry_list)]
            return df_new


    # 关联上公司基本信息
    def link_company_info(self, df):
        return pd.merge(df, self.company_info, on=['corp_org_id'], how='left')

