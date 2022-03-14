import preprocessing as prepro
import keyinfo_extraction as ke
from datetime import datetime
import topic_cluster
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib



# 交叉维度统计分析
class cross_analysis(object):

    # dimension代表维度的字段名。例如从客户维度输入consumer，从时间维度输入time，从行业维度输入industry
    def __init__(self, df, group_by, time_col_name):
        self.df = df.copy(deep=True)
        self.df[time_col_name] = self.df[time_col_name].apply(lambda x: x.split()[0])
        self.prepro = prepro.preprocess()
        self.word_count = ke.word_count()
        self.key_info_extraction = ke.keyinfo()
        self.dimension = group_by
        self.time_col_name = time_col_name

    # duration代表时间颗粒度，type代表选择的功能：词频统计、关键词提取
    def analyze_keyinfo(self, duration = "day", type = "count", algo = "tfidf", keyword_num = 10):

        if self.dimension == self.time_col_name:
            # 将df的多行合并
            if duration == "day":
                tc = self.df.groupby([self.time_col_name])["content"].apply(list).apply(self.list_combine)

            if duration == "week":
                self.df["time"] = self.df[self.time_col_name].apply(self.time_calculate, style=1)
                tc = self.df.groupby(["time"])["content"].apply(list).apply(self.list_combine)

            if duration == "month":
                self.df["time"] = self.df[self.time_col_name].apply(self.time_calculate, style=2)
                tc = self.df.groupby(["time"])["content"].apply(list).apply(self.list_combine)
        else:
            # 将df的多行合并
            if duration == "day":
                tc = self.df.groupby([self.dimension, self.time_col_name])["content"].apply(list).apply(self.list_combine)

            if duration == "week":
                self.df["time"] = self.df[self.time_col_name].apply(self.time_calculate, style=1)
                tc = self.df.groupby([self.dimension, "time"])["content"].apply(list).apply(self.list_combine)

            if duration == "month":
                self.df["time"] = self.df[self.time_col_name].apply(self.time_calculate, style=2)
                tc = self.df.groupby([self.dimension, "time"])["content"].apply(list).apply(self.list_combine)

        if type == "count":
            tc = tc.apply(self.word_count.count_word)
            return tc

        if type == "keyword":
            tc = tc.apply(" ".join)
            tc = tc.apply(self.key_info_extraction.jieba_keyword, keyword_num=keyword_num, algo=algo)
            return tc

    def analyze_topic(self, duration="day", number_of_clusters = 3):

        if duration == "day":
            tc = self.df.groupby([self.dimension, self.time_col_name])

        if duration == "week":
            self.df["time"] = self.df[self.time_col_name].apply(self.time_calculate, style=1)
            tc = self.df.groupby([self.dimension, "time"])

        if duration == "month":
            self.df["time"] = self.df[self.time_col_name].apply(self.time_calculate, style=2)
            tc = self.df.groupby([self.dimension, "time"])

        df_final = pd.DataFrame()
        for key, value in tc:
            value["content"] = value["content"].apply(" ".join)
            if len(value) < 3:
                num = 1
            else:
                num = number_of_clusters
            z = topic_cluster.topic_cluster(value)
            tfidf_train, word_dict = z.tfidf_vector()
            x = z.cluster_kmeans(tfidf_train, word_dict, num_clusters=num)
            df_final = pd.concat([df_final, x])

        return df_final

    def analyze_risk(self, duration = "day", risk_col_name = "risk_type_name"):

        if duration == "day":
            tc = self.df.groupby([self.dimension, self.time_col_name])[risk_col_name].apply(list)
            risk_count = tc.apply(self.risk_count)

        if duration == "week":
            self.df["time"] = self.df[self.time_col_name].apply(self.time_calculate, style=1)
            tc = self.df.groupby([self.dimension, "time"])[risk_col_name].apply(list)
            risk_count = tc.apply(self.risk_count)

        if duration == "month":
            self.df["time"] = self.df[self.time_col_name].apply(self.time_calculate, style=2)
            tc = self.df.groupby([self.dimension, "time"])[risk_col_name].apply(list)
            risk_count = tc.apply(self.risk_count)

        return risk_count

    def draw_risk(self, df, company_name, risk_col_name = "risk_type_name"):
        matplotlib.rcParams['font.sans-serif'] = ['SimHei']
        df = df[company_name].reset_index()
        risk_dicts = df[risk_col_name].tolist()
        risk_name_list = []

        for risk_dict in risk_dicts:
            risk_name_list = risk_name_list + list(risk_dict.keys())
        risk_name_set = set(risk_name_list)

        value = []
        for risk in risk_name_set:
            value.append(0)
            for risk_dict in risk_dicts:
                value[-1] += risk_dict.get(risk, 0)

        risk_name_list = list(risk_name_set)
        risk_name_list[0] = "正常舆情"

        fig, ax = plt.subplots(figsize=(5, 3), dpi=200)

        # 画柱状图
        width = 0.35  # the width of the bars: can also be len(x) sequence
        ax.set_xlabel('风险类型', fontsize=5)
        ax.set_ylabel('数量', fontsize=5)
        ax.set_title(company_name + "舆情风险", fontsize=5)
        plt.xticks(fontsize=5)
        ax.bar(risk_name_list, value, width)

        # # 画饼状图
        # ax.pie(value, labels=risk_name_list, radius=0.8,  # data 是数据，labels 是标签，radius 是饼图半径
        #        autopct='%1.1f%%',  # 显示所占比例，百分数
        #        pctdistance=0.5,
        #        labeldistance=0.7,  # a,b,c,d 到圆心的距离
        #        )  # 标签和比例的格式

        plt.show()


    def risk_count(self, risk_list):

        risk_dict = {}
        for risk in risk_list:
            risk_dict[risk] = risk_dict.get(risk, 0) + 1

        return risk_dict


    # 辅助函数用于时间字段的拆解
    def time_calculate(self, time, style):
        # 切分字符可能要更改
        time_split = time.split("/")
        year = int(time_split[0])
        month = int(time_split[1])
        date = int(time_split[2])
        end = int(datetime(year, month, date).strftime("%W"))
        begin = int(datetime(year, month, 1).strftime("%W"))
        week = end - begin + 1

        if style == 1:
            return str(year) + "年" + str(month) + "月" + "第" + str(week) + "周"
        if style == 2:
            return str(year) + "年" + str(month) + "月"

    # 辅助函数用于多列表的拼接
    def list_combine(self, ls):
        ls_new = []
        for i in ls:
            ls_new += i

        return ls_new
