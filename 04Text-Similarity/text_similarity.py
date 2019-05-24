#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import jieba
import numpy as np
import sys
import logging
import logging.handlers

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer


class cosSim:
    def __init__(self, logger=None):
        if logger is None:
            self.logger = logging.getLogger('cos_sim')
            self.logger.setLevel(logging.DEBUG)
            # 设置日志
            formater = logging.Formatter('%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')
            fh = logging.StreamHandler()  # 输出到终端
            fh.setLevel(level=logging.DEBUG) # error以上的内容输出到文件里面
            fh.setFormatter(formater)
            self.logger.addHandler(fh)

            rh = logging.handlers.TimedRotatingFileHandler('log.txt', when='D', interval=1, backupCount=30)
            rh.setLevel(level=logging.INFO)
            rh.setFormatter(formater)
            rh.suffix = "%Y%m%d_%H%M%S"
            self.logger.addHandler(rh)
        else:
            self.logger = logger

    def cos_sim(self, vector_a, vector_b):
        """
        计算两个向量之间的余弦相似度
        :param vector_a: 向量 a
        :param vector_b: 向量 b
        :return: sim
        """
        vector_a = np.mat(vector_a)
        vector_b = np.mat(vector_b)
        num = float(vector_a * vector_b.T)
        denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
        cos = num / denom
        sim = 0.5 + 0.5 * cos
        return sim

    def getVocabulary(self, corpuss):
        # max_features = 9000
        vectorizer = CountVectorizer()  # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
        transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值
        tfidf = transformer.fit_transform(
            vectorizer.fit_transform(corpuss))  # 第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
        words = vectorizer.get_feature_names()  # 获取词袋模型中的所有词语
        # self.logger.debug("words %s", words)
        return words

    def getVector(self, corpus, vocabulary):
        # self.logger.debug("corpus %s", corpus)
        vectorizer = CountVectorizer(vocabulary=vocabulary)  # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
        transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值
        # self.logger.debug("tf矩阵 %s", vectorizer.fit_transform(corpus))
        tfidf = transformer.fit_transform(
            vectorizer.fit_transform(corpus))  # 第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
        weight = tfidf.toarray()  # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重

        # weight = sorted(weight[0], reverse=True)
        # self.logger.debug("weight %s", weight)
        return weight

    def CalcuSim(self, texts=[]):
        """
        @:param list 需要对比的文本
        """
        if len(texts) != 2:
            raise Exception("texts长度必须为2")
        corpuss = [" ".join(jieba.cut(text)) for text in texts]
        vocabulary = self.getVocabulary(corpuss)
        v = self.getVector(corpuss, vocabulary=vocabulary)
        return self.cos_sim(v[0], v[1])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("error usage")
    cosSim = cosSim()
    texts = [item for item in sys.argv[1:]]
    texts = [
        '中国工商银行至尊皇家信用卡',
        '中国建设银行至尊皇家信用卡',
        "情况介绍：开学初，班上的一位较调皮的同学在某一天的课间，因为玩的太投入了，几乎忘乎所以，终于把我们班的一张凳子一分为二。而就在几天前也有一张凳子被分体了，关于这样的事情我已经详细的在班上进行了分析，我们是不可为的。同样的错误短短几天重复上演，我很生气，那位同学当然也很紧张。得知这个消息后，我就立即把她的家长喊过来了。在家长的全力配合下，该生很快认识到了自己的问题。于是我当着他妈妈的面对他说了很多语重心长的话，其中有一句是：“如果有什么意见，什么问题，尽管说，我能帮你解决的一定解决。”本以为老师说了这样的话后，学生一般都会说“没问题”，哪知道这位仁兄一点不客气的来了句“我去年犯错误，你让我做操站最后一个，今年我又没有犯错误，为什么还站在最后一个？”顿时，无语。本以为他是个高个子男生，站在最后面做操也没什么关系的，哪知道他一直以为这是对他的惩罚的延续。当然经过交流误会解除了，但这么一句话提醒了我，身为班主任要足够的细心，拥有敏锐的观察力，用心和每一位同学相处。案例分析：认真做事只能把事情做对，用心做事才能把事情做好。班主任工作确实很繁琐、细微，很多时候确实会很累，面对着学生们一遍又一遍的犯同一个错误，有时可能会真的失去耐心，但我们不能因此忽略对每一个学生的关爱，尤其是给容易犯错误的同学刻上烙印。相反，我们一定要给予这部分同学多一些关心，关爱，多找他们谈话，更多的了解他们的学习情况和生活情况。当然，和家长的交流也是必不可少的。班主任的工作有时真的是微不足道的，你没有办法去改变一个人性格，观念，习惯，我们唯一能做的就是在孩子成长的道路上给予引导和指引，让他们少走些弯路。",
        "你让我做操站最后一个，今年我又没有犯错误，为什么还站在最后一个？”顿时，无语。本以为他是个高个子男生，站在最后面做操也没什么关系的，哪知道他一直以为这是对他的惩罚的延续。当然经过交流误会解除了，但这么一句话提醒了我，身为班主任要足够的细心，拥有敏锐的观察力，用心和每一位同学相处。案例分析：认真做事只能把事情做对，用心做事才能把事情做好。班主任工作确实很繁琐、细微，很多时候确实会很累，面对着学生们一遍又一遍的犯同一个错误，有时可能会真的失去耐心，但我们不能因此忽略对每一个学生的关爱，尤其是给容易犯错误的同学刻上烙印。相反，我们一定要给予这部分同学多一些关心，关爱，多找他们谈话，更多的了解他们的学习情况和生活情况。当然，和家长的交流也是必不可少的。班主任的工作有时真的是微不足道的，你没有办法去改变一个人性格，观念，习惯，我们唯一能做的就是在孩子成长的道路上给予引导和指引，让他们少走些弯路。",

        "七（2）班德育案例（杨静）发布人:  时间：2013/6/25 9:54:11班主任工作的核心是德育工作，德育工作中最令班主任头痛的是转化后进生，转化后进生是老师所肩负的重大而艰巨的任务，也是教育工作者不容推卸的责任。我班有个学生叫龚坤。这学期前段时间，上课有时候会扰乱他人学习，要么不够认真；下课胡乱打闹，同学间经常闹矛盾，同学们都嫌弃他；本身成绩也不太好……经常科任老师或者学生向我告状。在一开始，真让我头痛。于是，我找他谈话，希望他在学校遵守各项规章制度，以学习为重，自我调节，自我控制，争取让自己的成绩有所进步。但经过几次努力，他只在口头上答应，行动上却毫无改进。尽管找他谈话时态度也比较恭敬，但转身后依然我行我素。看到他不为所动，我的心都快凉了，算了吧，或许他就是那根“不可雕的朽木”。不理他的那几天，他依然如故，小事频繁不断！ 此时，我觉得逃避不了，必须正视现实！我内心一横：我不改变你，誓不罢休！为了有针对性地做工作，我决定与其家长联系，进行详细了解，然后再找对策。通过其父亲的介绍，我才了解到：原来他一直就是这样，比较爱动，好玩。学习成绩不好，对学习失去兴趣。 在交谈之后，我内心久久不能平静，像打翻了的五味瓶！于是，转化他的行动在悄然中进行。我首先设法接近他，清除隔阂，拉近关系。经过观察，我发现他对于老师布置的劳动任务倒是比较积极。所以我经常与其沟通，要想实现自己的目标，一定要能坚定意志，要大家对你认同，而不是自己以为。要与同学搞好关系，不要与同学小摩擦，要做到学习上也能自觉，逐步改变你身上的小毛病。并提示他多参加有益的文体活动，这样对身体有好处。通过几次的接触，我与他慢慢交上了朋友,但他的纪律等并无多大改进。后来，我便加强攻势：一边与他交流讨论学习，进而讨论纪律。不动声色地教他遵守纪律，尊敬师长，团结同学，努力学习，做一名好学生。使他处处感到老师在关心他,信赖他。他也逐渐明白了做人的道理,明确了学习的目的。通过半学期的努力，他上课开始认真起来，与同学之间的关系也改善了，各科任老师都觉得他懂事了。由于纪律表现有所好转，尽管学习成绩没有得到明显的提高，但大家都对他能刮目相看了！",
    ]
    corpuss = [" ".join(jieba.cut(text)) for text in texts]
    vocabulary = cosSim.getVocabulary(corpuss)
    v = cosSim.getVector(corpuss, vocabulary=vocabulary)
    print(cosSim.cos_sim(v[2], v[3]))

    # VectorA = cosSim.getVector([corpuss[0]], vocabulary)
    # VectorB = cosSim.getVector([corpuss[1]], vocabulary)
    # print(cosSim.cos_sim(VectorA, VectorB))