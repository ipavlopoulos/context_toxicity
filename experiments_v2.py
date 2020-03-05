from sklearn.model_selection import train_test_split
import pandas as pd
from absl import flags, app
import numpy as np
import classifiers
from sklearn.metrics import *
from scipy.stats import sem
import tensorflow as tf
import os, sys
import datetime

FLAGS = flags.FLAGS
flags.DEFINE_string("model_name", None, "cc, ci, ch, ca, none, bert, bert_sep, bert_cc")  # name , default, help
flags.DEFINE_integer("with_context_data", 0, "False for context-less training data.")  # name , default, help
flags.DEFINE_integer("repeat", 0, "Repetitions of the experiment. Default is 0.")
flags.DEFINE_integer("epochs", 100, "Epochs. Default is 100.")
flags.DEFINE_string("data", "data/cctk_c.csv.zip", "The dataset dataframe to work with.")
flags.DEFINE_float("learning_rate", 1e-3, "Learning rate. Default is 1e-3.")
flags.DEFINE_integer("patience", 3, "Waiting epochs for the best performance. Default is 10.")  # name , default, help
flags.DEFINE_integer("seed", 42, "The seed to initialise the random state. Default is 42.")
flags.DEFINE_integer("verbose", 1, "Default is 1.")
flags.DEFINE_string("experiment_version_name", f"version-{datetime.datetime.now().strftime('%d%B%Y-%H%M')}", "The name of this series of experiments.")
flags.DEFINE_string("split", "622", "'811' for 8/1/1 split or '622' for 6/2/2 split")
flags.DEFINE_integer("aug_voc", 1, "Augmented vocabulary (using the parent comment) or not. Default is True.")

RNN = "none"
RNN_IN = "ci"
RNN_ATT = "ca"
RNN_HIE = "ch"
RNN_CC = "cc"
BERT = "bert"
BERT_SEP = "bert_sep"
BERT_CC = "bert_cc"

def get_rnn(seed):
    if FLAGS.model_name.lower() == RNN:
        print("Training RNN context-less.")
        rnn = classifiers.Rnn
    elif FLAGS.model_name.lower() == RNN_CC:
        print("Training RNN with context input simply concatenated to the target input.")
        rnn = classifiers.Rnn
    elif FLAGS.model_name.lower() == RNN_HIE:
        print("Training RNN with context (represented through another RNN) concatenated with the target before the "
              "top FFNN.")
        rnn = classifiers.RnnCh
    elif FLAGS.model_name.lower() == RNN_IN:
        print("Training RNN with context concatenated with word embeddings.")
        rnn = classifiers.RnnCi
    elif FLAGS.model_name.lower() == RNN_ATT:
        print("Training RNN with context used to attend the target encoding.")
        rnn = classifiers.RnnCa
    else:
        sys.exit("ERROR: Not implemented yet.\nContinuing with plain RNN.")
    return rnn(learning_rate=FLAGS.learning_rate, name=FLAGS.model_name.lower(), verbose=FLAGS.verbose,
                n_epochs=FLAGS.epochs, patience=FLAGS.patience, augmented_vocabulary=FLAGS.aug_voc, seed=seed)

def get_bert(seed, sess):
    lr = 2e-05
    os.environ['TFHUB_CACHE_DIR'] = 'embeddings'
    if FLAGS.model_name.lower() == BERT:
        print("Training BERT with no context mechanism added.")
        model = classifiers.BERT_MLP(patience=FLAGS.patience, seed=seed, lr=lr, epochs=FLAGS.epochs, session=sess)
    elif FLAGS.model_name.lower() == BERT_SEP:
        print("Training BERT with parent concatenated to text with [SEP] to separate the two.")
        model = classifiers.BERT_MLP(patience=FLAGS.patience, lr=lr, seed=seed, DATA2_COLUMN="parent", epochs=FLAGS.epochs, session=sess)
    elif FLAGS.model_name.lower() == BERT_CC:
        print("Training BERT with parent concatenated to text omitting the [SEP] separation token")
        model = classifiers.BERT_MLP(patience=FLAGS.patience, seed=seed, lr=lr, epochs=FLAGS.epochs, session=sess)
    else:
        print(f"ERROR: {FLAGS.model_name.lower()} is not implemented yet")
        model = None
    return model


def train(seed, sess=None):

    # load the data
    data_pd = pd.read_csv(FLAGS.data)
    test_size = 0.1 if FLAGS.split == "811" else 0.2
    train_pd, test_pd = train_test_split(data_pd, test_size=test_size, random_state=seed)
    train_pd, dev_pd = train_test_split(train_pd, test_size=test_pd.shape[0], random_state=seed)
    if "cc" in FLAGS.model_name.lower():
        for dataset in (train_pd, dev_pd, test_pd):
            dataset.text = dataset.parent.apply(lambda x: x + " ") + dataset.text
    print("Loading the embeddings...")
    embeddings = classifiers.load_embeddings_index()

    # build the model
    print("Creating the model...")
    if "bert" in FLAGS.model_name.lower():
        model = get_bert(seed, sess)
    elif "bert" not in FLAGS.model_name.lower():
        model = get_rnn(seed)

    # train it
    print(f"Training {model.name}...")
    model.fit(train=train_pd, dev=dev_pd, pretrained_embeddings=embeddings)

    # evaluate
    print("Evaluating...")
    predictions = model.predict(test_pd).flatten()
    score = roc_auc_score(test_pd.label.to_numpy(), predictions)
    print(f"ROC-AUC: {score}")
    print(f"STATS: toxicity (%) at predicted: {np.mean(predictions)} vs at gold: {test_pd.label.mean()}")
    return score, predictions, model


def repeat_experiment():
    scores = []
    predictions_pd = {}
    for i in range(FLAGS.repeat):
        with tf.compat.v1.Session() as sess:
            print(f"REPETITION: {i}")
            score, predictions, model = train(seed=i, sess=sess)
            scores.append(score)
            predictions_pd[i] = predictions
            model_name = model.name
    return np.mean(scores), sem(scores), predictions_pd, model_name


def main(argv):
    if FLAGS.repeat == 0:
        # Run at a single split
        score, predictions, model = train(FLAGS.seed)
        model_name = model.name
        print(f"{score}")
    elif FLAGS.repeat > 0:
        # Perform Monte Carlo Cross Validation
        score, sem, predictions, model_name = repeat_experiment()
        print (f"{score} ± {sem}")
    save_path = f"{FLAGS.experiment_version_name}.{model_name}_predictions.csv"
    pd.DataFrame(predictions).to_csv(save_path, index=False)


if __name__ == "__main__":
    app.run(main)

