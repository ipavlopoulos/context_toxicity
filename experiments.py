from sklearn.model_selection import train_test_split
import pandas as pd
from absl import flags, logging, app
import numpy as np
import helper
import classifiers
from sklearn.metrics import *
from scipy.stats import sem
import tensorflow as tf
import os, sys
import datetime

FLAGS = flags.FLAGS
flags.DEFINE_string("with_context_model", None,
                    "'OOC' for context-less architecture or IC{1/2/3} for context-aware architectures.")  # name , default, help
flags.DEFINE_boolean("with_context_data", False, "False for context-less training data.")  # name , default, help
flags.DEFINE_integer("oversample", 1, "Oversample the positive class, e.g., 99/1 (enter 99)")
flags.DEFINE_integer("repeat", 0, "Repetitions of the experiment. Default is 0.")
flags.DEFINE_integer("epochs", 100, "Epochs. Default is 100.")
flags.DEFINE_boolean("confidence_intervals", False, "Show Confidence Intervals along with AUROCC")  # name , default, help
flags.DEFINE_integer("create_random_splits", 0, "Create random splits. Default number is 0, which means: 'do not split'.")
flags.DEFINE_integer("patience", 10, "Waiting epochs for the best performance. Default is 10.")  # name , default, help
flags.DEFINE_integer("seed", 42, "The seed to initialise the random state. Default is 42.")
flags.DEFINE_boolean("create_balanced_datasets", False, "If True, use downsampling to create balanced versions of the original datasets.")
flags.DEFINE_boolean("use_balanced_datasets", False, "If True, use the balanced datasets.")
flags.DEFINE_string("splits_version", "random_ten", "The name of the splits directory. Default is 'random_ten'.")
flags.DEFINE_string("experiment_version_name", f"version-{datetime.datetime.now().strftime('%d%B%Y-%H%M')}", "The name of the splits directory. Default is 'standard_ten'.")

def create_balanced_datasets():
    """
    Create balanced versions of the original datasets. Positive (here, toxic) examples have been removed from
    the less imbalanced class, while the same number of negative examples have been removed from the more
    imbalanced class. The outcome is two equally sized and balanced datasets.
    :return:
    """
    oc_pd = pd.read_csv(f"data/original/oc.csv")
    oc_pd = oc_pd.sample(frac=1, random_state=FLAGS.seed)
    ic_pd = pd.read_csv(f"data/original/wc.csv")
    ic_pd = ic_pd.sample(frac=1, random_state=FLAGS.seed)
    print(f"Class balance of datasets, InC vs. OoC: {oc_pd.label.sum()/oc_pd.shape[0]} - {ic_pd.label.sum()/ic_pd.shape[0]}")
    # remove positive (toxic) examples from the less imbalanced set and negative from the other
    diff = ic_pd.label.sum() - oc_pd.label.sum()
    ic_pd.drop(ic_pd[ic_pd.label == 1].index[:diff], inplace=True)
    oc_pd.drop(oc_pd[oc_pd.label == 0].index[:diff], inplace=True)
    print(f"InC vs. OoC: {oc_pd.label.sum()/oc_pd.shape[0]} - {ic_pd.label.sum()/ic_pd.shape[0]}")
    os.mkdir("data/balanced")
    ic_pd.to_csv("data/balanced/wc.csv")
    oc_pd.to_csv("data/balanced/oc.csv")

def split_to_random_sets(splits=10, schema="standard", version_name="random_ten"):
    """
    Split the datasets to random sets.
    :param splits: Number of sets to split.
    :param schema: The type of the original datasets. Note that the type should exist in the filepath.
    :param version: The version name, under which the splits will be saved.
    :return:
    """
    assert schema in {"standard", "balanced"}
    path_name = f"data/{schema}/{version_name}"
    if os.path.exists(path_name):
        sys.exit(f"ERROR: {path_name} is not empty.")
    os.makedirs(path_name)
    for split_num in range(splits):
        os.makedirs(f"{path_name}/{split_num}")
        for setting in ("wc", "oc"):
            data_pd = pd.read_csv(f"data/{schema}/{setting}.csv")
            train_pd, val_pd = train_test_split(data_pd, test_size=0.1, random_state=FLAGS.seed)
            train_pd, dev_pd = train_test_split(train_pd, test_size=val_pd.shape[0], random_state=FLAGS.seed)
            train_pd.to_csv(f"{path_name}/{split_num}/{setting.replace('w','i')}.train.csv")
            dev_pd.to_csv(f"{path_name}/{split_num}/{setting.replace('w','i')}.dev.csv")
            val_pd.to_csv(f"{path_name}/{split_num}/{setting.replace('w','i')}.val.csv")

def train(with_context, model_setting, verbose=1, splits_path="data/standard/splits-10", the_split_to_use=0):
    print(f"Loading the data... Using the '{splits_path}/{the_split_to_use}' split.")
    if with_context:
        train_pd = pd.read_csv(f"{splits_path}/{the_split_to_use}/ic.train.csv")
        dev_pd = pd.read_csv(f"{splits_path}/{the_split_to_use}/ic.dev.csv")
    else:
        train_pd = pd.read_csv(f"{splits_path}/{the_split_to_use}/oc.train.csv")
        dev_pd = pd.read_csv(f"{splits_path}/{the_split_to_use}/oc.dev.csv")
    val_pd = pd.read_csv(f"{splits_path}/{the_split_to_use}/ic.val.csv")

    print("Loading the embeddings...")
    class_weights = {0: 1, 1: FLAGS.oversample}
    embeddings = classifiers.load_embeddings_index()

    print("Creating the model...")
    with tf.Session() as sess:
        if model_setting == "RNN:OOC":
            model = classifiers.LSTM_CLF(prefix=model_setting.lower(), verbose=verbose, n_epochs=FLAGS.epochs)
        else:
            if model_setting == "RNN:INC1":
                model = classifiers.LSTM_IC1_CLF(prefix=model_setting.lower(), verbose=verbose,  n_epochs=FLAGS.epochs)
            elif model_setting == "RNN:INC2":
                model = classifiers.LSTM_IC2_CLF(prefix=model_setting.lower(), verbose=verbose,  n_epochs=FLAGS.epochs)
            elif "RNN" in model_setting:
                print("Not implemented yet...")
            else:
                if "BERT" in model_setting:
                    os.environ['TFHUB_CACHE_DIR'] = 'embeddings'
                    lr = 2e-05
                    patience = FLAGS.patience
                    if model_setting == "BERT:OOC1":
                        print("Training BERT with no context mechanism added.")
                        model = classifiers.BERT_MLP(patience=patience, lr=lr,  epochs=FLAGS.epochs, session=sess)
                    elif model_setting == "BERT:OOC2":
                        print("Training BERT with text-parent concatenated.")
                        model = classifiers.BERT_MLP(patience=patience, lr=lr, DATA2_COLUMN="parent", epochs=FLAGS.epochs, session=sess)
                    elif model_setting == "BERT:OOC3":
                        print("Training BERT with parent-text concatenated.")
                        model = classifiers.BERT_MLP(patience=patience, lr=lr, DATA2_COLUMN="parent", epochs=FLAGS.epochs, session=sess)
                    elif model_setting == "BERT:CA":
                        print("Training BERT with a context-reading mechanism added.")
                        model = classifiers.BERT_MLP_CA(patience=patience, lr=lr, epochs=FLAGS.epochs, session=sess)
                    else:
                        sys.exit("ERROR: Not implemented yet...")

        print("Training...")
        model.fit(train=train_pd, dev=dev_pd, class_weights=class_weights, pretrained_embeddings=embeddings)
        gold, predictions = val_pd.label.to_numpy(), model.predict(val_pd).flatten()
        score = roc_auc_score(gold, predictions)
        print("Evaluating...")
        print(f"STATS: toxicity (%) at predicted: {np.mean(predictions)} vs at gold: {np.mean(gold)}")
        if not FLAGS.confidence_intervals:
            print(f"ROC-AUC: {score}")
        else:
            score, intervals = helper.CIB(gold_truth=list(gold), predictions=list(predictions)).evaluate()
            print(f"ROC-AUC ± CIs: {score} ± {intervals}")
    return score, predictions, model


def repeat_experiment(with_context, model_setting, steps):
    scores = []
    predictions_pd = pd.DataFrame()
    model_name = ""
    splits_path = f"data/{'balanced' if FLAGS.use_balanced_datasets else 'standard'}/{FLAGS.splits_version}"
    if not os.path.exists(splits_path):
        sys.exit(f"ERROR: {splits_path} is empty! Make sure the desired dataset is successfully created.")
    if FLAGS.use_balanced_datasets:
        FLAGS.experiment_version_name += ".balanced"
    os.mkdir(FLAGS.experiment_version_name)
    for i in range(steps):
        print(f"REPETITION: {i}")
        score, predictions, model = train(with_context, model_setting, splits_path=splits_path, the_split_to_use=i)
        scores.append(score)
        predictions_pd[f"split{i}"] = predictions
        model_name = model.name
    return np.mean(scores), sem(scores), predictions_pd, model_name # the last model used - the same for all runs

def main(argv):
    if FLAGS.create_random_splits>0:
        print(f"Splitting the data randomly into {FLAGS.create_random_splits} splits")
        schema = "balanced" if FLAGS.use_balanced_datasets else "standard"
        split_to_random_sets(splits=FLAGS.create_random_splits, schema=schema, version_name=FLAGS.splits_version)
    elif FLAGS.create_balanced_datasets:
        create_balanced_datasets()
    else:
        score, sem, predictions_pd, model_name = repeat_experiment(with_context=FLAGS.with_context_data, model_setting=FLAGS.with_context_model, steps=FLAGS.repeat)
        predictions_pd.to_csv(f"{FLAGS.experiment_version_name}/{model_name}.predictions.csv")
        print (f"{score} ± {sem}")

if __name__ == "__main__":
    app.run(main)
