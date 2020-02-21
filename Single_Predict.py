import tensorflow as tf
import collections
import tokenization
import modelling
import os
import csv

tf.logging.set_verbosity(tf.logging.INFO)

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer("train_batch_size", 20, "Total batch size for training.")
flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")
flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")
flags.DEFINE_integer("num_train_epochs", 5, "Total epoches for train.")
flags.DEFINE_string("data_dir", "./datasets",
                    "The input data dir. Should contain the .tsv files (or other data files) for the task.")
flags.DEFINE_string("init_checkpoint", None, "Initial checkpoint")
flags.DEFINE_string("vocab_file", "./vocab.txt", "The vocabulary file.")
flags.DEFINE_string("model_dir", "./new_model", "The output file for trained model.")

flags.DEFINE_bool("do_train", True, "Whether to run training.")
flags.DEFINE_bool("do_eval", True, "Whether to run eval on the dev set.")
flags.DEFINE_bool("do_predict", False, "Whether to run the model in inference mode on the test set. ")
flags.DEFINE_bool("do_single_predict", False, "Whether to run the model in inference mode on the single test.")


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, label=None):
        """Constructs a InputExample."""
        self.guid = guid
        self.text = text
        self.label = label


class InputFeatures(object):
    def __init__(self, input_ids, input_size):
        self.input_ids = input_ids
        self.input_size = input_size
        # self.label = label


class DataProcessor(object):

    def get_train_examples(self, data_dir):
        lines = self._read_tsv(os.path.join(data_dir, "online_shopping_train.tsv"))
        return self._create_examples(lines, 'train')

    def get_dev_examples(self, data_dir):
        lines = self._read_tsv(os.path.join(data_dir, "online_shopping_dev.tsv"))
        return self._create_examples(lines, 'dev')

    def get_test_examples(self, data_dir):
        lines = self._read_tsv(os.path.join(data_dir, "online_shopping_test.tsv"))
        return self._create_examples(lines, 'test')

    @staticmethod
    def get_labels():
        return ["0", "1"]
        # return ['蒙牛', '水果', '洗发水', '平板', '酒店', '手机', '计算机', '书籍', '衣服', '热水器']

    @staticmethod
    def _create_examples(lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text = tokenization.convert_to_unicode(line[3])
            label = tokenization.convert_to_unicode(line[2])
            examples.append(
                InputExample(guid=guid, text=text, label=label))
        return examples

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


# 将一个example类的训练数据转成feature类
def convert_single_example(ex_index, example, tokenizer):
    text = example.text
    tokens = tokenizer.tokenize(text)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_size = len(input_ids)
    try:
        label = int(example.label)
    except TypeError:
        label = None
    # 打印前5条转换的记录
    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % example.guid)
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_size: %s" % input_size)
        tf.logging.info("label: %s" % label)
    feature = InputFeatures(input_ids=input_ids, input_size=input_size)
    return feature


# 将准备喂入模型的数据存成tfrecord文件
def file_based_convert_examples_to_features(examples, tokenizer, output_file):
    writer = tf.python_io.TFRecordWriter(output_file)
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))
        feature = convert_single_example(ex_index, example, tokenizer)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features['input_ids'] = create_int_feature(feature.input_ids)
        features['input_size'] = create_int_feature([feature.input_size])
        # features['label'] = create_int_feature([feature.label])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()


def file_based_input_fn_builder(input_file, is_training, batch_size, num_epochs):
    def parse_func(serialized_example):
        name_to_features = {
            "input_ids": tf.VarLenFeature(tf.int64),
            "input_size": tf.FixedLenFeature(shape=(1,), dtype=tf.int64)
            # "label": tf.FixedLenFeature(shape=(1,), dtype=tf.int64),
        }
        parsed_example = tf.parse_single_example(serialized_example, features=name_to_features)
        parsed_example['input_ids'] = tf.sparse_tensor_to_dense(parsed_example['input_ids'])

        input_ids = parsed_example['input_ids']
        input_size = parsed_example['input_size']
        # label = parsed_example['label']

        return input_ids, input_size

    def input_fn():

        dataset = tf.data.TFRecordDataset(input_file)
        dataset = dataset.map(parse_func)

        if is_training:
            dataset = dataset.repeat(num_epochs).shuffle(buffer_size=100)

        padded_shapes = (tf.TensorShape([None]),  # 语料数据，None即代表batch_size
                         tf.TensorShape([None]))  # 语料数据各个句子的原始长度

        # 调用padded_batch方法进行batching操作
        batched_dataset = dataset.padded_batch(batch_size, padded_shapes)

        iterator = batched_dataset.make_one_shot_iterator()
        features = iterator.get_next()
        features = {
            'input_ids': features[0],
            'input_sizes': tf.reshape(features[1], shape=(-1,))
        }
        return features
    return input_fn


def model_fn_builder(config):
    def model_fn(features, labels, mode):
        input_ids = features['input_ids']
        input_sizes = features['input_sizes']
        # 构建模型，得到lstm结构的输出结果
        model = modelling.LstmModel(config=config,
                                    mode=mode,
                                    input_ids=input_ids,
                                    input_sizes=input_sizes)
        logits = model.logits

        probabilities = tf.nn.softmax(logits, axis=-1)

        predictions = {
            'classes': tf.argmax(logits, axis=1),
            'probabilities': probabilities
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        # 损失函数
        with tf.variable_scope('loss'):
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                                  logits=logits)
            total_loss = tf.reduce_mean(loss)

        if mode == tf.estimator.ModeKeys.TRAIN:
            trainable_variables = tf.trainable_variables()

            # # 控制梯度大小，定义优化方法和训练步骤
            grads = tf.gradients(total_loss, trainable_variables)
            grads, _ = tf.clip_by_global_norm(grads, config.max_grad_norm)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=config.learning_rate)
            train_op = optimizer.apply_gradients(
                zip(grads, trainable_variables),
                global_step=tf.train.get_global_step())
            # train_op = optimizer.minimize(total_loss, global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=total_loss, train_op=train_op)

        if mode == tf.estimator.ModeKeys.EVAL:
            # 准确率
            with tf.variable_scope('accuracy'):
                # correct_prediction = tf.equal(tf.argmax(logits, 1), labels)
                # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions['classes'])
            eval_metric_ops = {"eval_accuracy": accuracy}
            return tf.estimator.EstimatorSpec(mode=mode, loss=total_loss, eval_metric_ops=eval_metric_ops)
    return model_fn


def single_predict(text_data):
    test_data = []
    for index in range(len(text_data)):
        guid = 'test-%d' % index
        text = tokenization.convert_to_unicode(str(text_data[index]))
        # label = str(test[2])
        test_data.append(InputExample(guid=guid, text=text, label=None))

    predict_examples = test_data
    num_actual_predict_examples = len(predict_examples)

    config = modelling.LstmConfig(vocab_size=68355)
    tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file)

    model_fn = model_fn_builder(config)
    tf.gfile.MakeDirs(FLAGS.model_dir)
    estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=FLAGS.model_dir)

    predict_file = r'./tmp/predict.tfrecord'
    file_based_convert_examples_to_features(predict_examples, tokenizer, predict_file)

    predict_input_fn = file_based_input_fn_builder(predict_file,
                                                   is_training=False,
                                                   batch_size=FLAGS.predict_batch_size,
                                                   num_epochs=None)

    result = estimator.predict(input_fn=predict_input_fn)
    res_dic = {}
    num_written_lines = 0
    for (k, prediction) in enumerate(result):
        predict_result = prediction['classes']
        probabilities = prediction["probabilities"]
        print(predict_result, max(probabilities))
        print('------------------')


if __name__ == '__main__':
    sentences = ['交通方便；环境很好；服务态度很好 房间较小上',
                 '太差了，空调的噪音很大，设施也不齐全，携程怎么会选择这样的合作伙伴',
                 '哈弗F7的车机太糟心了，花这么多钱就买了这么一个烂玩意',
                 '一句普通中立的话']
    single_predict(sentences)
