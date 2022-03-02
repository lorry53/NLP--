#!/usr/bin/env python
# coding: utf-8


"""
已实现功能如下：
keyinfo_extraction：关键词、关键短语、关键句抽取
news_sentiment：评价新闻是正面、负面还是中性的（待完善）
cross_analysis：提供从时间、战略客户、行业维度交叉统计分析的功能
word_count：词频统计、词云
company_relationship：公司关系图谱生成（待解决）
rank_importance：新闻重要性排序
"""


# 导入各处理模块
import pandas as pd
import preprocessing as pp
import cross_analysis as ca
import filter_duplicate as fd
import keyinfo_extraction as ke
import relationship_map as rm
import topic_cluster as tc
import doc_rank as dc


"""
预处理模块：完成解析、分词等文本清洗工作
"""
# 读取头条新闻
p = pp.preprocess()
df = p.read_doc("data_source/头条数据.csv", False)
# 解析html。读入头条新闻doc_type="title_news"，公告新闻doc_type="pubnote"
df["content"] = df["content"].apply(p.html_encoding, doc_type = "title_news")
# 标记需要删除的文本
df["del"] = df["content"].apply(p.del_check)
# 筛选出保留的文本
df = df[df["del"] == 0]
df = df.drop(columns="del")

# 筛选功能，type=0按战略客户名称筛选，type=1按日期筛选（需要提供start_time和end_time），type=2按行业筛选（提供行业清单）
df = pp.preprocess().filter_df(df, 0)

# 文本分词与清洗
df["ori_content"] = df["content"]
df["content"] = df['content'].apply(p.clean_data)

# 和客户信息表关联（只要公告表和新闻表需要关联）
df = pp.preprocess().link_company_info(df)

# 导出结果
df.to_csv("data_output/公告结果.csv")



"""
关键信息提取模块：提供关键词提取、关键句提取、词频统计、词云生成四项功能
"""
# 关键词获取
# 关键词个数
keyword_num = 10
df["keyword"] = df["content"].apply(ke.keyinfo().jieba_keyword, keyword_num=keyword_num)
print(df["keyword"])

# 关键句获取（此时不需要进行分词）
# 关键句个数
keysentence_num = 3
df["keysentence"] = df["content"].apply(ke.keyinfo().textrank_keysentence, keysentence_num=keysentence_num)

# 词频统计
print(ke.word_count().count_word(df["content"].iloc[1]))

# 词云绘制
ke.word_count().wordcloud(df["content"].iloc[1])


"""
交叉分析模块：交叉分析：时间维度输入time、客户维度输入consumer、行业维度输入industry
"""
# duration代表时间颗粒度，type代表选择的功能：词频统计count或关键词提取keyword。当提取关键词时，有两种算法可供选择algo=textrank/tfidf
x = ca.cross_analysis(df, "consumer").analyze(duration = "week", type = "count", algo="tfidf", keyword_num=20)
print(x)


"""
公司关系图谱模块：公司之间用线段相连，线段颜色代表联系紧密程度
"""
# 公司关系权重图获取
rm.company_relationship(df).creat_relationship()


"""
新闻去重模块：标记内容重复的新闻，提供公司颗粒度的去重
"""
# 新闻去重
sim = fd.SimHash()
df["simhash"] = df["content"].apply(sim.simHash)
df_final = sim.filter_duplicate(df)

# 如希望从公司的维度去重，采用以下命令
df = df.groupby(["full_name"])
df_final = pd.DataFrame()
for key, value in df:
    x = sim.filter_duplicate(value)
    df_final = pd.concat([df_final, x])
df_final.to_csv("data_output/去重结果_头条xin.csv")



"""
新闻聚类模块：按主题将文本聚类，提供新闻所属类别的标记，以及每一类别的关键词集合
"""
# 新闻聚类
df["content"] = df["content"].apply(" ".join)
z = tc.topic_cluster(df)
tfidf_train, word_dict = z.tfidf_vector()
z.best_kmeans(tfidf_train)
df = z.cluster_kmeans(tfidf_train, word_dict, num_clusters=8)
df.to_csv("data_output/聚类结果.csv")



"""
新闻重要性排序+时间轴生成模块
"""
# 新闻重要性排序+时间轴
handler = dc.Docrank()
result = handler.doc_graph()
timelines = handler.timeline(result)



