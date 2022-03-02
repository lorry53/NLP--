import preprocessing as prepro
import keyinfo_extraction as ke
from datetime import datetime



# 交叉维度统计分析
class cross_analysis(object):

    # dimension代表维度的字段名。例如从客户维度输入consumer，从时间维度输入time，从行业维度输入industry
    def __init__(self, df, dimension):
        self.df = df.copy(deep=True)
        self.df["created_at"] = self.df["created_at"].apply(lambda x: x.split()[0])
        self.prepro = prepro.preprocess()
        self.word_count = ke.word_count()
        self.key_info_extraction = ke.keyinfo()
        if dimension == "consumer":
            self.dimension = "full_name"
        if dimension == "time":
            self.dimension = "created_at"
        if dimension == "industry":
            self.dimension = "industry"

    # duration代表时间颗粒度，type代表选择的功能：词频统计或关键词提取
    def analyze(self, duration = "day", type = "count", algo = "tfidf", keyword_num = 10):

        if self.dimension == "created_at":
            # 将df的多行合并
            if duration == "day":
                tc = self.df.groupby(["created_at"])["content"].apply(list).apply(self.list_combine, type = type)

            if duration == "week":
                self.df["time"] = self.df["created_at"].apply(self.time_calculate, type=1)
                tc = self.df.groupby(["time"])["content"].apply(list).apply(self.list_combine, type = type)

            if duration == "month":
                self.df["time"] = self.df["created_at"].apply(self.time_calculate, type=2)
                tc = self.df.groupby(["time"])["content"].apply(list).apply(self.list_combine, type = type)
        else:
            # 将df的多行合并
            if duration == "day":
                tc = self.df.groupby([self.dimension, "created_at"])["content"].apply(list).apply(self.list_combine, type = type)

            if duration == "week":
                self.df["time"] = self.df["created_at"].apply(self.time_calculate, type=1)
                tc = self.df.groupby([self.dimension, "time"])["content"].apply(list).apply(self.list_combine, type = type)

            if duration == "month":
                self.df["time"] = self.df["created_at"].apply(self.time_calculate, type=2)
                tc = self.df.groupby([self.dimension, "time"])["content"].apply(list).apply(self.list_combine, type = type)

        if type == "count":
            tc = tc.apply(self.word_count.count_word)

        if type == "keyword":
            tc.apply(" ".join)
            tc = tc.apply(self.key_info_extraction.jieba_keyword, keyword_num=keyword_num, algo=algo)

        return tc

    # 辅助函数用于created_at字段的拆解
    def time_calculate(self, time, type):
        time_split = time.split("-")
        year = int(time_split[0])
        month = int(time_split[1])
        date = int(time_split[2])
        end = int(datetime(year, month, date).strftime("%W"))
        begin = int(datetime(year, month, 1).strftime("%W"))
        week = end - begin + 1

        if type == 1:
            return str(year) + "年" + str(month) + "月" + "第" + str(week) + "周"
        if type == 2:
            return str(year) + "年" + str(month) + "月"

    # 辅助函数用于多列表的拼接
    def list_combine(self, ls, type):
        ls_new = []
        for i in ls:
            ls_new += i

        return ls_new