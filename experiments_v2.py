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
flags.DEFINE_string("model_name", None, "rnn _cc, _ci, _ch, _ca, _none, _chl, _chl2; bert, bert _sep, _cc, _hl;")  # name , default, help
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
flags.DEFINE_string("parent_label", "toxicity_parent", "The label of the parent comment.")

RNN = "rnn_none"
RNN_IN = "rnn_ci"
RNN_ATT = "rnn_ca"
RNN_HIE = "rnn_ch"
RNN_HL = "rnn_chl"
RNN_HL2 = "rnn_chl2"
RNN_CC = "rnn_cc"
BERT = "bert"
BERT_SEP = "bert_sep"
BERT_CC = "bert_cc"
BERT_HL = "bert_hl"
BERT_HLG = "bert_hl_gold"
B1 = "baseline_perspective_MLP"
B2 = "baseline_perspective@parent"
B3 = "baseline_perspective@target"


def get_baseline(seed):
    if FLAGS.model_name.lower() == B1:
        print("Training MLP with Perspective scores.")
        return classifiers.MlpH(learning_rate=FLAGS.learning_rate,
                     name=FLAGS.model_name.lower(),
                     verbose=FLAGS.verbose,
                     n_epochs=FLAGS.epochs,
                     patience=FLAGS.patience,
                     seed=seed)
    elif FLAGS.model_name.lower() == B2:
        return classifiers.Baseline(label="toxicity_parent")
    elif FLAGS.model_name.lower() == B3:
        return classifiers.Baseline(label="perspective_target")
    else:
        sys.exit("ERROR: NOT IMPLEMENTED YET!")


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
    elif FLAGS.model_name.lower() == RNN_HL:
        print("Training RNN with parent comment representation and label concatenated to the target"
              "comment representation before the top FFNN.")
        rnn = classifiers.RnnChl
        return rnn(learning_rate=FLAGS.learning_rate, parent_lbl=FLAGS.parent_label, name=FLAGS.model_name.lower(), verbose=FLAGS.verbose,
                   n_epochs=FLAGS.epochs, patience=FLAGS.patience, augmented_vocabulary=FLAGS.aug_voc, seed=seed)
    elif FLAGS.model_name.lower() == RNN_HL2:
        print("Parent and Target scored by Perspective and parsed by RNN")
        rnn = classifiers.RnnChl2
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
    elif FLAGS.model_name.lower() == BERT_HL:
        print("Training BERT with parent score integrated before the FFNN.")
        model = classifiers.BERT_FFNN_HL(patience=FLAGS.patience, seed=seed, lr=lr, epochs=FLAGS.epochs, session=sess)
    elif FLAGS.model_name.lower() == BERT_HLG:
        print("Training BERT with parent GOLD score integrated before the FFNN.")
        model = classifiers.BERT_FFNN_HL(use_gold_parent=True, patience=FLAGS.patience, seed=seed, lr=lr, epochs=FLAGS.epochs, session=sess)
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

    # some exceptions
    if "cc" in FLAGS.model_name.lower():
        for dataset in (train_pd, dev_pd, test_pd):
            dataset.text = dataset.parent.apply(lambda x: x + " ") + dataset.text
    print("Loading the embeddings...")
    embeddings = classifiers.load_embeddings_index()

    # build the model
    print("Creating the model...")
    if "baseline" in FLAGS.model_name.lower():
        model = get_baseline(seed)
    elif "bert" in FLAGS.model_name.lower():
        model = get_bert(seed, sess)
    elif "rnn" in FLAGS.model_name.lower():
        model = get_rnn(seed)
    else:
        sys.exit("ERROR: NOT IMPLEMENTED YET.")

    # train it
    print(f"Training {model.name}...")
    model.fit(train=train_pd, dev=dev_pd, pretrained_embeddings=embeddings)

    # evaluate
    print("Evaluating...")
    predictions = model.predict(test_pd).flatten()
    roc = get_roc(test_pd, predictions)
    return roc, predictions, model


def get_roc(test, predictions):
    score = roc_auc_score(test.label.to_numpy(), predictions)
    print(f"ROC-AUC: {score}")
    print(f"STATS: toxicity (%) at predicted: {np.mean(predictions)} vs at gold: {test.label.mean()}")
    return score


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

