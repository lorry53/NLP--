import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx



# 生成公司关系图谱
class company_relationship(object):

    def __init__(self, df, name_list=None, type = 0):
        self.df_copy = df.copy(deep = True)
        self.competitor_name = []

        # 获取战略客户列表
        self.client_name = pd.read_csv("data_source/客户全称&简称.csv")
        if name_list==None:
            self.name_dict = dict(zip(self.client_name["全称"].tolist(), self.client_name["简称"].tolist()))
        else:
            self.name_dict = name_list

        with open(r"data_source/竞争对手.txt", "r", encoding='UTF-8') as f:
            for line in f.readlines():
                self.competitor_name.append(line.strip('\n'))

        # 公司关系图谱字典
        self.relations = {}

        # type = 0绘制客户之间关系图谱，type=1绘制客户进而竞争对手之间关系图谱
        self.type = type

        matplotlib.rcParams['font.sans-serif'] = ['SimHei']


    # 简称向全称转换
    def short_to_full(self, text):
        return [list(self.name_dict.keys())[list(self.name_dict.values()).index(word)] if word in self.name_dict.values() and word != "nan" else word for word in text]

    def relation_count(self, text):
        if len(text) == 0:
            return
        if self.type == 0:
            names = self.name_dict.keys()
        elif self.type == 1:
            names = self.competitor_name

        for name_0 in self.name_dict.keys():
            if name_0 in text:
                for name_1 in names:
                    if name_1 in text and name_0 != name_1 and (name_1, name_0) not in self.relations:
                        self.relations[(name_0, name_1)] = self.relations.get((name_0, name_1), 0) + 1

    # 生成公司关系图
    def creat_relationship(self):

        self.df_copy["content"] = self.df_copy["content"].apply(self.short_to_full)
        self.df_copy["content"].apply(self.relation_count)

        # 画公司关系权重图
        maxRela = max([v for k, v in self.relations.items()])
        relations = {k: v / maxRela for k, v in self.relations.items()}

        plt.figure(figsize=(30, 30))
        G = nx.Graph()
        for k, v in relations.items():
            G.add_edge(k[0], k[1], weight=v)
        # 筛选权重大于0.6的边
        elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] > 0.6]
        # 筛选权重大于0.2小于0.6的边
        emidle = [(u, v) for (u, v, d) in G.edges(data=True) if (d['weight'] > 0.2) & (d['weight'] <= 0.6)]
        # 筛选权重小于0.2的边
        esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] <= 0.2]
        # 设置图形布局
        pos = nx.spring_layout(G, k=0.6,iterations=20)
        # 设置节点样式
        nx.draw_networkx_nodes(G, pos, alpha=0.8, node_size=500)
        # 设置大于0.6的边的样式
        nx.draw_networkx_edges(G, pos, edgelist=elarge, width=2.5, alpha=1, edge_color='g')
        # 0.2~0.6
        nx.draw_networkx_edges(G, pos, edgelist=emidle, width=1.5, alpha=1, edge_color='y')
        # <0.2
        nx.draw_networkx_edges(G, pos, edgelist=esmall, width=1, alpha=0.1, edge_color='b', style='dashed')
        nx.draw_networkx_labels(G, pos, font_size=5)


        plt.axis('off')
        plt.title("公司关系图谱")
        plt.savefig("data_output/公司关系图谱.jpg")
        plt.show()