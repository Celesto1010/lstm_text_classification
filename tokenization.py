"""
Tokenization，基本取自Bert源码，做了相当的精简
"""


import collections
import tensorflow as tf
import jieba


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    else:
        raise ValueError("Unsupported string type: %s" % (type(text)))


# 将词典中的词构成(词，index)的collections.OrderedDict形式
def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    with tf.gfile.GFile(vocab_file, "r") as reader:
        while True:
            token = convert_to_unicode(reader.readline())
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab


def convert_by_vocab(vocab, items):
    """Converts a sequence of [tokens|ids] using the vocab."""
    output = []
    for item in items:
        output.append(vocab.get(item, vocab['[UNK]']))
    return output


class FullTokenizer(object):
    """Runs end-to-end tokenziation."""

    def __init__(self, vocab_file):
        # 根据vocab文件，得到形如(词，index)的字典
        self.vocab = load_vocab(vocab_file)
        # 变成 index: 词 的形式
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    # 将句子变成词列表
    @staticmethod
    def tokenize(text):
        split_tokens = jieba.lcut(text)
        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        return convert_by_vocab(self.vocab, tokens)

    def convert_ids_to_tokens(self, ids):
        return convert_by_vocab(self.inv_vocab, ids)


# tokenizer = FullTokenizer(r'./vocab.txt')
# tokens = tokenizer.tokenize('测试一下这个能不能用魑魅魍')
# print(tokens)
# ids = tokenizer.convert_tokens_to_ids(tokens)
# print(ids)