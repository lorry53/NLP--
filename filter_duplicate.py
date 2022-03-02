import jieba.analyse
import numpy as np



class SimHash(object):

    def __init__(self):
        pass

    def simHash(self, tokenized_text):

        # jieba基于TF-IDF提取关键词
        jieba.analyse.set_stop_words('data_source/停用词.txt')
        keyWords = jieba.analyse.extract_tags('|'.join(tokenized_text), topK=20, withWeight=True, allowPOS=())

        keyList = []
        for feature, weight in keyWords:
            weight = int(weight*20)
            # 获取哈希值
            binstr = self.string_hash(feature)
            temp=[]
            # 加权
            for c in binstr:
                if (c == '1'):
                    temp.append(weight)
                else:
                    temp.append(-weight)
            keyList.append(temp)
        # 合并
        listSum = np.sum(np.array(keyList), axis = 0)
        if (keyList == []):
            return '00'
        # 降维
        simhash = ''
        for i in listSum:
            if (i>0):
                simhash = simhash + '1'
            else:
                simhash = simhash + '0'

        return simhash


    # 哈希生成函数
    def string_hash(self, source):
        if source == "":
            return 0
        else:
            x = ord(source[0]) << 7
            m = 1000003
            mask = 2**128 - 1
            for c in source:
                x = ((x*m)^ord(c)) & mask
            x ^= len(source)
            if x == -1:
                x = -2
            x = bin(x).replace('0b', '').zfill(64)[-64:]

            return str(x)

    # 计算两个simhash的汉明距离
    def getDistance(self, hashstr1, hashstr2):
        length = 0
        for index, char in enumerate(hashstr1):
            if char == hashstr2[index]:
                continue
            else:
                length += 1

        return length

    # 过滤重复新闻
    def filter_duplicate(self, df):
        mark = 1
        df["duplicate"] = 0

        for i in range(len(df["simhash"])):
            hashstr1 = df["simhash"].iloc[i]
            for j in range(i+1, len(df["simhash"])):
                hashstr2 = df["simhash"].iloc[j]
                if self.getDistance(hashstr1, hashstr2)<3:
                    if df["duplicate"].iloc[i] != 0:
                        df["duplicate"].iloc[j] = df["duplicate"].iloc[i]
                    else:
                        df["duplicate"].iloc[i] = mark
                        df["duplicate"].iloc[j] = mark
                        mark += 1

        return df