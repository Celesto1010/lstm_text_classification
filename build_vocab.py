"""
本文件定义OnlineShopping语料的预处理。包括：
1. 从csv文件读入语料数据，切词，将切好词的语料数据与标签数据单独存储
2. 给单词表添加PADDING与UNK，将单词表单独存储（vocab.txt文件）
3. 以单词编号的形式，单独存储切好词的语料数据
"""

import jieba
import pandas as pd


# 原始的文件
RAW_DATA = r'./datasets/online_shopping_10_cats.csv'
STOPWORDS = r'E:/NLP/字典/停用词词典/stopwords_snownlp.txt'

# vocab文件与编码文件
VOCAB = r'./vocab.txt'


# 基于本任务的数据得到vocab文件
def create_vocab():

    raw_df = pd.read_csv(RAW_DATA)                          # 读原始文件为dataframe
    # 热水器有一条数据有问题，不要热水器的数据
    raw_df = raw_df[raw_df.cat != '热水器']

    raw_document = raw_df['review'].tolist()                # 原始语料（list形式）

    # 加载停用词列表
    # with open(STOPWORDS, 'r', encoding='utf-8') as s:
    #     stopwords = [word.strip() for word in s.readlines()]

    document_words = []                                     # 原始语料完成切词
    for sentence in raw_document:
        cut_sentence = [word for word in jieba.lcut(sentence)]
        document_words.extend(cut_sentence)
    vocab_list = set(document_words)

    with open(VOCAB, 'w', encoding='utf-8') as f:
        f.write('[PAD]' + '\n')
        f.write('[UNK]' + '\n')
        for vocab in vocab_list:
            f.write(vocab + '\n')


if __name__ == '__main__':
    create_vocab()
