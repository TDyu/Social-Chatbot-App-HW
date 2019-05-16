#!/user/bin/env python3
# -*- coding: utf-8 -*-
"""关心于不同关键字/景点时，标题/评论的情绪极性、大家讨论的/感受的方向局势（词云）、相似度（分别使用TF-IDF模型&LSI模型）
*使用数据解释：本来是要用期末专题中的景点评论数据来做作业，但是爬虫被封了（x）还没爬到什么，只好先硬是用在作业2收集的数据上，虽然应用在标题上有点奇怪*
*需配合档案：
    数据来源的三个档案(GNZ_search_article_informations.csv，SNH_search_article_informations.csv，BEJ_search_article_informations.csv)、
    字体档(msyh.ttf)、
    停用词(hgd_stopwords.txt)、
    情感辞典(BosonNLP_sentiment_score.txt)、
    否定词(notDict.txt)、
    程度副词辞典(degreeDict.txt)
"""
from hanziconv import HanziConv  # 简繁转换
import pandas as pd
import jieba
import jieba.posseg as pseg
import codecs
from gensim import corpora, models, similarities
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import re


def _traditional_to_simple_chinese(text):
    """繁体转做简体会好处理一些，因为辞典都用简体的
    """
    if type(text) == str:
        return HanziConv.toSimplified(text)
    elif type(text) == list:
        return [HanziConv.toSimplified(line) for line in text ]


def _break_word(statement):
    """Break a statement to a list include its words.

    Args:
        statement: str, Target statement.
    
    Returns:
        A list representation of the words in statement.

    Notes:
        https://github.com/fxsjy/jieba
    """
    stopwords_path = './hgd_stopwords.txt'

    # Load the stop words.
    with open(stopwords_path, 'r', encoding='utf8') as f:
        stops = f.read().split('\n')

    # Default mode is precise mode.
    # segments_list = list(jieba.cut(
    #     statement, cut_all=False))
    # And remove stopwords.
    segments_list = list(word for word in jieba.cut(
        statement, cut_all=False) if word not in stops)

    return segments_list


def _cut_word_and_organize(text_list):
    """对于一“群”文本句子要把分词整理在一起时的处理
    """
    # 先将每一个文本转成简体且分词 并且去除停留词
    remove_break_word_text_list = [_break_word(
        _traditional_to_simple_chinese(text)) for text in text_list]
    # print(remove_break_word_text_list)

    # 再将个别文本的分词句串起来，成为一个“主题”的分词，准备生成这个主题的词云
    # 顺便移除空白格
    text_cut_list = []
    for single_cut_list in remove_break_word_text_list:
        for cut in single_cut_list:
            if cut != ' ':
                text_cut_list.append(cut)
    # print(text_cut_list)
    return text_cut_list


def generate_wordcloud(text_list, output_path=None):
    """对要分析的文本群产生词云
    应用场景：
        对作业2的PPT标题：一眼知道大家讨论什么局势状况
        对期末专题的景点评论：一眼知道大家对景点感受的大部分的状况
    """
    # 将其每一个转成简体且做分词，并且整体串成一个主题下的分词
    text_cut_list = _cut_word_and_organize(text_list)

    # 为了解决要使用二进制才能用wordcloud的转换
    with open('./text_cut.txt', 'w') as f:
        for cut in text_cut_list:
            f.write(cut)
            f.write(' ')
    with open('./text_cut.txt', 'r') as f:
        text_cut_list = f.read()

    # 生成词云
    wordcloud = WordCloud(
        #设置字体，不然会出现口字乱码
        font_path='msyh.ttf',
        #设置了背景，宽高
        background_color="white",
        width=1000,
        height=880).generate(text_cut_list)
    # 若有设置output_path则保存图片，没有则直接显示
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()
    plt.clf()


def calculate_text_similarity(target_text_list, target_text_list2, query_text_list):
    """计算相似度，分别用TF-IDF模型和LSI模型
    应用场景：
        对作业2的PPT标题：比较一个关键字对上另外俩关键字下的所有标题的相似度，若是关键字是城市，就能知道一个城市对上另外俩城市讨论的事情相似度
        对期末专题的景点评论：比较两个评论相似度，可以用来更加确定可信度，或是用来剔除刷评论的
    """
    # 将其每一个转成简体且做分词，并且整体串成一个主题下的分词
    target_text_cut_list = _cut_word_and_organize(target_text_list)
    target_text_cut_list2 = _cut_word_and_organize(target_text_list2)
    corpus = [target_text_cut_list, target_text_cut_list2]

    # 建立词袋模型
    dictionary = corpora.Dictionary(corpus)
    # print(dictionary)
    doc_vectors = [dictionary.doc2bow(text) for text in corpus]
    # print(len(doc_vectors))
    # print(doc_vectors)

    # 建立TF-IDF模型
    tfidf = models.TfidfModel(doc_vectors)
    tfidf_vectors = tfidf[doc_vectors]
    # print(len(tfidf_vectors))
    # print(len(tfidf_vectors[0]))

    # 构建一个query文本
    query_text_cut_list = _cut_word_and_organize(query_text_list)
    query_bow = dictionary.doc2bow(query_text_cut_list)
    # print(len(query_bow))
    # print(query_bow)
    index = similarities.MatrixSimilarity(tfidf_vectors)

    # 用TF-IDF模型计算相似度(语料较少的情况下，效果不高)
    sims = index[query_bow]
    sims_index_list = list(enumerate(sims))
    # print(sims_index_list)  # [(0, 0.06642722), (1, 0.06336182)]
    print('相似度 - TF-IDF模型：')
    print('对于第一个的相似度：', str(sims_index_list[0][1]))
    print('对于第二个的相似度：', str(sims_index_list[1][1]))

    # 构建LSI模型，设置主题数为2
    lsi = models.LsiModel(tfidf_vectors, id2word=dictionary, num_topics=2)
    lsi.print_topics(2)
    lsi_vector = lsi[tfidf_vectors]
    # for vec in lsi_vector:
        # print(vec)
    query_lsi = lsi[query_bow]
    # print(query_lsi)
    index = similarities.MatrixSimilarity(lsi_vector)
    sims = index[query_lsi]
    sims_index_list = list(enumerate(sims))
    # print(sims_index_list)  # [(0, 0.06642722), (1, 0.06336182)]
    print('相似度 - LSI模型：')
    print('对于第一个的相似度：', str(sims_index_list[0][1]))
    print('对于第二个的相似度：', str(sims_index_list[1][1]))


def _classify_words(word_dict):
    """情感定位，将句子中各类词分别存储并标注位置
    """
    # 读取情感字典文件
    sen_file = open('./BosonNLP_sentiment_score.txt', 'r+', encoding='utf-8')
    # 获取字典文件内容
    sen_list = sen_file.readlines()
    # 创建情感字典
    sen_dict = defaultdict()
    # 读取字典文件每一行内容，将其转换为字典对象，key为情感词，value为对应的分值
    for s in sen_list:
        # 去除\n & 空白行
        # 每一行内容根据空格分割，索引0是情感词，索引01是情感分值
        s_split = s.split(' ')
        if len(s_split) < 2:
                continue
        if '\n' in s_split[1]:
            s_split[1] = s_split[1].replace('\n', '')
        sen_dict[s_split[0]] = s_split[1]

    # 读取否定词文件
    not_word_file = open('./notDict.txt', 'r+', encoding='utf-8')
    # 由于否定词只有词，没有分值，使用list即可
    not_word_list = not_word_file.readlines()

    # 读取程度副词文件
    degree_file = open('./degreeDict.txt', 'r+', encoding='utf-8')
    degree_list = degree_file.readlines()
    degree_dic = defaultdict()
    # 程度副词与情感词处理方式一样，转为程度副词字典对象，key为程度副词，value为对应的程度值
    for d in degree_list:
        degree_dic[d.split(',')[0]] = d.split(',')[1]

    # 分类结果，词语的index作为key,词语的分值作为value，否定词分值设为-1
    sen_word = dict()
    not_word = dict()
    degree_word = dict()

    # 分类
    for word in word_dict.keys():
        if word in sen_dict.keys() and word not in not_word_list and word not in degree_dic.keys():
            # 找出分词结果中在情感字典中的词
            sen_word[word_dict[word]] = sen_dict[word]
        elif word in not_word_list and word not in degree_dic.keys():
            # 分词结果中在否定词列表中的词
            not_word[word_dict[word]] = -1
        elif word in degree_dic.keys():
            # 分词结果中在程度副词中的词
            degree_word[word_dict[word]] = degree_dic[word]
    sen_file.close()
    degree_file.close()
    not_word_file.close()
    # 将分类结果返回
    return sen_word, not_word, degree_word


def _get_init_weight(sen_word, not_word, degree_word):
    # 权重初始化为1
    W = 1
    # 将情感字典的key转为list
    sen_word_index_list = list(sen_word.keys())
    if len(sen_word_index_list) == 0:
        return W
    # 获取第一个情感词的下标，遍历从0到此位置之间的所有词，找出程度词和否定词
    for i in range(0, sen_word_index_list[0]):
        if i in not_word.keys():
            W *= -1
        elif i in degree_word.keys():
            # 更新权重，如果有程度副词，分值乘以程度副词的程度分值
            W *= float(degree_word[i])
    return W


def _socre_sentiment(sen_word, not_word, degree_word, seg_result):
    """情感聚合
    简化的情感分数计算逻辑：所有情感词语组的分数之和
    定义一个情感词语组：两情感词之间的所有否定词和程度副词与这两情感词中的后一情感词构成一个情感词组，
    即not_words + degree_words + senti_words，例如不是很交好，其中不是为否定词，很为程度副词，交好为情感词，那么这个情感词语组的分数为：
    finalSentiScore = (-1) ^ 1 * 1.25 * 0.747127733968
    其中1指的是一个否定词，1.25是程度副词的数值，0.747127733968为交好的情感分数。
    """
    # 权重初始化为1
    W = 1
    score = 0
    # 情感词下标初始化
    sentiment_index = -1
    # 情感词的位置下标集合
    sentiment_index_list = list(sen_word.keys())
    # 遍历分词结果(遍历分词结果是为了定位两个情感词之间的程度副词和否定词)
    for i in range(0, len(seg_result)):
        # 如果是情感词（根据下标是否在情感词分类结果中判断）
        if i in sen_word.keys():
            # 权重*情感词得分
            score += W * float(sen_word[i])
            # 情感词下标加1，获取下一个情感词的位置
            sentiment_index += 1
            if sentiment_index < len(sentiment_index_list) - 1:
                # 判断当前的情感词与下一个情感词之间是否有程度副词或否定词
                for j in range(sentiment_index_list[sentiment_index], sentiment_index_list[sentiment_index + 1]):
                    # 更新权重，如果有否定词，取反
                    if j in not_word.keys():
                        W *= -1
                    elif j in degree_word.keys():
                        # 更新权重，如果有程度副词，分值乘以程度副词的程度分值
                        W *= float(degree_word[j])
        # 定位到下一个情感词
        if sentiment_index < len(sentiment_index_list) - 1:
            i = sentiment_index_list[sentiment_index + 1]
    return score


def _list_to_dict(word_list):
    """将分词后的列表转为字典，key为单词，value为单词在列表中的索引，索引相当于词语在文档中出现的位置
    """
    data = {}
    for x in range(0, len(word_list)):
        data[word_list[x]] = x
    return data


def analy_text_sentiment_polarity(text_list):
    """计算情绪极性
    应用场景：
        对作业2的PPT标题：比较一个关键字所有标题的正负情绪极性
        对期末专题的景点评论：对于一个景点的正负情绪极性
    """
    # 将其每一个转成简体且做分词，并且整体串成一个主题下的分词
    text_cut_list = _cut_word_and_organize(text_list)

    # 将分词后的列表转为字典
    word_dict = _list_to_dict(text_cut_list)

    # 找出情感词、否定词、程度副词
    sen_word, not_word, degree_word = _classify_words(word_dict)

    # 计算得分
    score = _socre_sentiment(sen_word, not_word, degree_word, text_cut_list)
    print(score)
    return score


if __name__ == "__main__":
    # 1. 读入所需档案
    # "index_col=0" to get rid of `Unnamed:` column in a dataframe
    gnz_text_list = list(pd.read_csv(
        './GNZ_search_article_informations.csv', index_col=0)['title'])
    snh_text_list = list(pd.read_csv(
        './SNH_search_article_informations.csv', index_col=0)['title'])
    bej_text_list = list(pd.read_csv(
        './BEJ_search_article_informations.csv', index_col=0)['title'])
    # print(gnz_text_list)

    # 2. 产生词云
    # 产生关键字是“广州”时的标题的词云
    generate_wordcloud(gnz_text_list, output_path=None)
    # 产生关键字是“上海”时的标题的词云
    generate_wordcloud(snh_text_list, output_path=None)
    # 产生关键字是“北京”时的标题的词云
    generate_wordcloud(bej_text_list, output_path=None)

    # 3. 计算一个城市对上另外俩城市文章标题相似度
    calculate_text_similarity(
        snh_text_list, bej_text_list, gnz_text_list)

    # 4. 计算情绪极性分数
    analy_text_sentiment_polarity(gnz_text_list)
