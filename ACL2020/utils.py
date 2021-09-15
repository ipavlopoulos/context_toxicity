import collections
import csv
import tensorflow as tf
from sklearn.metrics import *
import pandas as pd
import numpy as np
from tensorflow.keras.callbacks import Callback
import logging
# Following is a dependency on the ssig package:
#! git clone https://github.com/ipavlopoulos/ssig.git
from ssig import art


def ca_perspective(n=5):
    """
    Evaluate PERSPECTIVE with parent-target concatenated.
    Scores provided to us.
    :param n:
    :return:
    """
    c = pd.read_csv("data/c_parenttext.csv")
    c.set_index(["id"], inplace=True)
    data = [pd.read_csv(f"data/standard.622/random_ten/{i}/ic.val.csv") for i in range(n)]
    scores = []
    for sample in data:
        sample["ca_score"] = sample["id"].apply(lambda x: c.loc[x].TOXICITY)
        scores.append(roc_auc_score(sample.label, sample.ca_score))
    return scores


def persp_vs_capersp(n=5):
    c = pd.read_csv("data/c_parenttext.csv")
    c.set_index(["id"], inplace=True)
    data = [pd.read_csv(f"data/standard.622/random_ten/{i}/ic.val.csv") for i in range(n)]
    val = pd.concat(data)
    val.drop_duplicates(["id"], inplace=True)
    val["ca_score"] = val["id"].apply(lambda x: c.loc[x].TOXICITY)
    ca_score = roc_auc_score(val.label, val.ca_score)
    baseline_score = roc_auc_score(val.label, val.api)
    p = art.compare_systems(gold=val.label.to_list(),
                            system_predictions=val.ca_score.to_list(),
                            baseline_predictions=val.api.to_list(),
                            evaluator=roc_auc_score)
    return ca_score, baseline_score, p


def rocauc(y_true, y_pred):
    return tf.cond(tf.reduce_max(y_true) == 1,
                   tf.keras.metrics.AUC(y_true, y_pred),
                   lambda x: 1.0
                   )


class CallbackAUC(Callback):
    def __init__(self, validation_data):
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.x_val)
        roc = roc_auc_score(self.y_val, y_pred)
        logging.info(f'\r -- roc-auc: {str(round(roc,4))}\n')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.
    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.
    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_id,
                 is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.is_real_example = is_real_example


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        return InputFeatures(
            input_ids=[0] * max_seq_length,
            input_mask=[0] * max_seq_length,
            segment_ids=[0] * max_seq_length,
            label_id=0,
            is_real_example=False)

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    tokens_a = tokenizer.tokenize(example.text_a)
    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)

    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    label_id = label_map[example.label]

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=label_id,
        is_real_example=True)
    return feature


def file_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenizer)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature([feature.label_id])
        features["is_real_example"] = create_int_feature(
            [int(feature.is_real_example)])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenizer)

        features.append(feature)
    return features


def perspective_evaluate(split=0, schema="standard", mode="val", setting="ic"):
    ic = pd.read_csv(f"data/{schema}/random_ten/{split}/{setting}.{mode}.csv")
    return roc_auc_score(ic.label, ic.api)

class PERSPECTIVE_WRAPPER():

    def __init__(self):
        from googleapiclient import discovery
        import config
        self.API_KEY = config.GOOGLE_API_KEY
        # Generates API client object dynamically based on service name and version.
        self.service = discovery.build('commentanalyzer', 'v1alpha1', developerKey=self.API_KEY)

    def call(self, text, lan=None, max_chars=2000):
        analyze_request = {
            'comment': {'text': text[:max_chars]},
            'requestedAttributes': {'TOXICITY': {}}
        }
        if lan is not None:
            analyze_request['languages'] = [lan]
        try:
            response = self.service.comments().analyze(body=analyze_request).execute()
            return response['attributeScores']['TOXICITY']['summaryScore']['value']
        except Exception as e:
            print('FAIL: %s' % str(e))
            return -1

    def batch_call(self, data):
        probs, fails = {}, []
        for i, d in enumerate(data):
            p = self.call(d)
            if p < 0:
                fails.append(i)
            else:
                probs[i] = p
        # WARNING: If attribute language is used by default in call_perspective, many texts are not fetched
        for i in fails:
            probs[i] = self.call(data[i], lan='en')
        return probs

    def evaluate_at_split(self, split=9, schema="standard", with_context=True):
        ic = pd.read_csv(f"data/{schema}/random_ten/{split}/ic.val.csv")
        texts = ic.text
        if with_context:
            texts = ic.parent + ic.text
        scores = self.batch_call(texts.to_list())
        return scores