import tensorflow as tf
import json
import six


class LstmConfig(object):

    def __init__(self,
                 vocab_size,  # 词典中的词数
                 hidden_size=128,
                 keep_prob=0.9,
                 embedding_keep_prob=0.9,  # 词向量不被dropout的比例
                 max_grad_norm=5,
                 num_of_classes=2,  # 分类数
                 num_of_layers=2,  # lstm网络层数
                 learning_rate=0.4,  # 学习率
                 initializer_range=0.02):  # 初始化范围
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.embedding_keep_prob = embedding_keep_prob
        self.max_grad_norm = max_grad_norm
        self.num_of_classes = num_of_classes
        self.num_of_layers = num_of_layers
        self.learning_rate = learning_rate
        self.initializer_range = initializer_range

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = LstmConfig(vocab_size=None)
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with tf.gfile.GFile(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))


# 双向LSTM网络模型
class LstmModel(object):

    # 构建网格结构
    def __init__(self, config, mode, input_ids, input_sizes):

        embedding_keep_prob = config.embedding_keep_prob
        output_keep_prob = config.keep_prob if mode == tf.estimator.ModeKeys.TRAIN else 1.0

        # 词向量
        self.word_embedding = tf.get_variable('word_emb', shape=[config.vocab_size, config.hidden_size])
        # 将输入的序号化单词转成词向量
        inputs = tf.nn.embedding_lookup(self.word_embedding, input_ids)
        if mode == tf.estimator.ModeKeys.TRAIN:
            inputs = tf.nn.dropout(inputs, embedding_keep_prob)

        # lstm网络结构
        # 前向网络变量
        lstm_cells_fw = [tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(config.hidden_size),
                                                       output_keep_prob=output_keep_prob)
                         for _ in range(config.num_of_layers)]
        self.lstm_fw = tf.nn.rnn_cell.MultiRNNCell(lstm_cells_fw)
        # 反向网络
        lstm_cells_bw = [tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(config.hidden_size),
                                                       output_keep_prob=output_keep_prob)
                         for _ in range(config.num_of_layers)]
        self.lstm_bw = tf.nn.rnn_cell.MultiRNNCell(lstm_cells_bw)

        # Softmax层变量
        self.weight = tf.get_variable('weight', [config.hidden_size * 2, config.num_of_classes],
                                      initializer=tf.truncated_normal_initializer(stddev=0.02))
        self.bias = tf.get_variable('bias', [config.num_of_classes], initializer=tf.zeros_initializer())

        # LSTM网络计算
        with tf.variable_scope('lstm'):
            outputs, states = tf.nn.bidirectional_dynamic_rnn(self.lstm_fw,
                                                              self.lstm_bw,
                                                              inputs,
                                                              dtype=tf.float32,
                                                              sequence_length=input_sizes)
            # final_outputs = tf.concat(outputs, 2)
            # final_outputs = final_outputs[:, -1, :]
            # 取平均值
            outputs = tf.reduce_mean(tf.concat(outputs, 2), 1)

        # 全连接层计算
        with tf.variable_scope('fc'):
            self.logits = tf.matmul(outputs, self.weight) + self.bias

    def get_lstm_output(self):
        return self.logits
