from sklearn.model_selection import train_test_split
import pandas as pd
from absl import flags, logging, app
import numpy as np
import classifiers
from sklearn.metrics import *
from scipy.stats import sem
import tensorflow as tf
import os, sys
import json
import datetime
# Following is a dependency on the ssig package:
#! git clone https://github.com/ipavlopoulos/ssig.git
from ssig import ci

FLAGS = flags.FLAGS
flags.DEFINE_string("model_name", None, "name:OOC' for Out Of Context architecture (or the respective context-aware schema).")  # name , default, help
flags.DEFINE_integer("with_context_data", 0, "False for context-less training data.")  # name , default, help
flags.DEFINE_integer("oversample", 1, "Oversample the positive class, e.g., 99/1 (enter 99)")
flags.DEFINE_integer("repeat", 0, "Repetitions of the experiment. Default is 0.")
flags.DEFINE_integer("at_split", 0, "Operate on specific split. Default is 0.")
flags.DEFINE_integer("epochs", 100, "Epochs. Default is 100.")
flags.DEFINE_integer("confidence_intervals", 1, "Show Confidence Intervals along with AUROCC")  # name , default, help
flags.DEFINE_integer("create_random_splits", 0, "Create random splits. Default number is 0, which means: 'do not split'.")
flags.DEFINE_integer("patience", 3, "Waiting epochs for the best performance. Default is 10.")  # name , default, help
flags.DEFINE_integer("seed", 42, "The seed to initialise the random state. Default is 42.")
flags.DEFINE_integer("create_balanced_datasets", 0, "If True, use downsampling to create balanced versions of the original datasets.")
flags.DEFINE_string("splits_version", "random_ten", "The name of the splits directory. Default is 'random_ten'.")
flags.DEFINE_string("experiment_version_name", f"version-{datetime.datetime.now().strftime('%d%B%Y-%H%M')}", "The name of the splits directory. Default is 'standard_ten'.")
flags.DEFINE_string("schema", "balanced", "'standard' for original distributions, 'balanced' for downsampled")
flags.DEFINE_string("split", "standard.622", "'standard' for 80/10/10 split, 'standard.622' for 60/20/20 split")
flags.DEFINE_string("bert_weights", None, "Load and use the weights of a pre-trained BERT.")

def evaluate_perspective(dataset_path="data/CAT_LARGE", splits=10):
    scores = []
    for i in range(splits):
        ic = pd.read_csv(f"{dataset_path}/{i}/ic.val.csv")
        scores.append(roc_auc_score(ic.label, ic.api))
    return scores

def create_balanced_datasets(path="data/CAT_LARGE/0"):
    """
    Create balanced versions of a dataset.
    Positive (here, toxic) examples have been removed from the less imbalanced class,
    while the same number of negative examples have been removed from the more
    imbalanced class. The outcome is two equally sized and balanced datasets.
    :return:
    """
    for mode in ("train", "val", "dev"):
        oc_pd = pd.read_csv(f"{path}/oc.{mode}.csv")
        oc_pd = oc_pd.sample(frac=1, random_state=FLAGS.seed)
        ic_pd = pd.read_csv(f"{path}/ic.{mode}.csv")
        ic_pd = ic_pd.sample(frac=1, random_state=FLAGS.seed)
        #print(f"Class balance of datasets, InC vs. OoC: {oc_pd.label.sum()/oc_pd.shape[0]} - {ic_pd.label.sum()/ic_pd.shape[0]}")
        # remove positive (toxic) examples from the less imbalanced set and negative from the other
        diff = ic_pd.label.sum() - oc_pd.label.sum()
        ic_pd.drop(ic_pd[ic_pd.label == 1].index[:diff], inplace=True)
        oc_pd.drop(oc_pd[oc_pd.label == 0].index[:diff], inplace=True)
        #print(f"InC vs. OoC: {oc_pd.label.sum()/oc_pd.shape[0]} - {ic_pd.label.sum()/ic_pd.shape[0]}")
        ic_pd.to_csv(f"{path}/ic.{mode}.balanced.csv")
        oc_pd.to_csv(f"{path}/oc.{mode}.balanced.csv")


def split_to_random_sets(splits=10, test_size=0.2):
    """
    Split the datasets to random sets.
    :param test_size:
    :param splits: Number of sets to split.
    :return:
    """
    path_name = f"data/CAT_LARGE"
    if os.path.exists(path_name):
        sys.exit(f"ERROR: {path_name} is not empty.")
    os.makedirs(path_name)
    for split_num in range(splits):
        os.makedirs(f"{path_name}/{split_num}")
        for setting in ("gc", "gn"):
            data_pd = pd.read_csv(f"data/{setting}.csv")
            train_pd, val_pd = train_test_split(data_pd, test_size=test_size, random_state=FLAGS.seed+split_num)
            train_pd, dev_pd = train_test_split(train_pd, test_size=val_pd.shape[0], random_state=FLAGS.seed+split_num)
            train_pd.to_csv(f"{path_name}/{split_num}/{setting}.train.csv", index=False)
            dev_pd.to_csv(f"{path_name}/{split_num}/{setting}.dev.csv", index=False)
            val_pd.to_csv(f"{path_name}/{split_num}/{setting}.val.csv", index=False)

def train(with_context, verbose=1, splits_path="data/CAT_LARGE", the_split_to_use=9):
    print(f"Loading the data... Using the '{splits_path}/{the_split_to_use}' split.")
    ctx_id = 'c' if with_context>0 else 'n'
    train_pd = pd.read_csv(f"{splits_path}/{the_split_to_use}/g{ctx_id}.train.csv")
    dev_pd = pd.read_csv(f"{splits_path}/{the_split_to_use}/g{ctx_id}.dev.csv")
    val_pd = pd.read_csv(f"{splits_path}/{the_split_to_use}/gc.val.csv")
    print (f"INFO: CNTXT: {ctx_id}")
    print("Loading the embeddings...")
    class_weights = {0: 1, 1: FLAGS.oversample}
    embeddings = classifiers.load_embeddings_index()

    print("Creating the model...")
    with tf.compat.v1.Session() as sess:
        if FLAGS.model_name == "RNN:OOC":
            model = classifiers.LSTM_CLF(prefix=FLAGS.model_name.lower(), verbose=verbose, n_epochs=FLAGS.epochs)
        else:
            if FLAGS.model_name == "RNN:INC1":
                model = classifiers.LSTM_IC1_CLF(prefix=FLAGS.model_name.lower(), verbose=verbose,  n_epochs=FLAGS.epochs, patience=FLAGS.patience)
            elif FLAGS.model_name == "RNN:INC2":
                model = classifiers.LSTM_IC2_CLF(prefix=FLAGS.model_name.lower(), verbose=verbose,  n_epochs=FLAGS.epochs, patience=FLAGS.patience)
            elif "RNN" in FLAGS.model_name:
                print("Not implemented yet...")
            else:
                if "BERT" in FLAGS.model_name:
                    os.environ['TFHUB_CACHE_DIR'] = 'embeddings'
                    lr = 2e-05
                    if FLAGS.model_name == "BERT:OOC":
                        print("Training BERT with no context mechanism added.")
                        model = classifiers.BERT_MLP(patience=FLAGS.patience, lr=lr,  epochs=FLAGS.epochs, session=sess)
                    elif FLAGS.model_name == "BERT:INC1":
                        print("Training BERT with parent concatenated to text.")
                        model = classifiers.BERT_MLP(patience=FLAGS.patience, lr=lr, DATA2_COLUMN="parent", epochs=FLAGS.epochs, session=sess)
                    elif FLAGS.model_name == "BERT:INC2":
                        print("Training BERT with a context-reading mechanism added.")
                        model = classifiers.BERT_MLP_CA(patience=FLAGS.patience, lr=lr, epochs=FLAGS.epochs, session=sess)
                    elif FLAGS.model_name == "BERT:CCTK":
                        print("Training BERT over CCTK")
                        model = classifiers.BERT_MLP(patience=FLAGS.patience, lr=lr, epochs=FLAGS.epochs, session=sess)
                        cctk = pd.read_csv("data/CCTK.csv.zip", nrows=100000)
                        x_train_pd, x_dev_pd = train_test_split(
                            pd.DataFrame({"text": cctk.comment_text, "label": cctk.target.apply(round)}),
                            test_size=0.1,
                            random_state=FLAGS.seed
                        )
                        model.fit(train=x_train_pd,
                                  dev=x_dev_pd,
                                  class_weights=class_weights,
                                  pretrained_embeddings=embeddings)
                        cctk_preds_pd = pd.DataFrame()
                        for i in range(10):
                            x_val_pd = pd.read_csv(f"data/CAT_LARGE/{i}/ic.val.csv")
                            gold, predictions = x_val_pd.label.to_numpy(), model.predict(x_val_pd).flatten()
                            score = roc_auc_score(gold, predictions)
                            print(f"ROC-AUC@{i}: {score}")
                            cctk_preds_pd[f"MCCV_{i}"] = predictions
                        cctk_preds_pd.to_csv("cctk.csv")
                        model.model.save_weights("bert_weights.h5")
                    else:
                        sys.exit("ERROR: Not implemented yet...")

        print(f"Training {model.name}...")
        if "BERT" in FLAGS.model_name:
            model.fit(train=train_pd, dev=dev_pd, bert_weights=FLAGS.bert_weights, class_weights=class_weights, pretrained_embeddings=embeddings)
        else:
            model.fit(train=train_pd, dev=dev_pd, class_weights=class_weights, pretrained_embeddings=embeddings)

        gold, predictions = val_pd.label.to_numpy(), model.predict(val_pd).flatten()
        score = roc_auc_score(gold, predictions)
        print("Evaluating...")
        print(f"ROC-AUC: {score}")
        print(f"STATS: toxicity (%) at predicted: {np.mean(predictions)} vs at gold: {np.mean(gold)}")
        if FLAGS.confidence_intervals !=0:
            score, intervals = ci.AUC(gold_truth=list(gold), predictions=list(predictions)).evaluate()
            print(f"ROC-AUC ± CIs: {score} ± {intervals}")
    return score, predictions, model


def repeat_experiment():
    scores = []
    predictions_pd = pd.DataFrame()
    model_name = ""
    splits_path = f"data/CAT_LARGE"
    if not os.path.exists(splits_path):
        sys.exit(f"ERROR: {splits_path} is empty! Make sure the desired dataset is successfully created.")
    FLAGS.experiment_version_name += "." + FLAGS.schema
    os.mkdir(FLAGS.experiment_version_name)
    for i in range(FLAGS.repeat):
        print(f"REPETITION: {i}")
        score, predictions, model = train(FLAGS.with_context_data, splits_path=splits_path, the_split_to_use=i)
        scores.append(score)
        predictions_pd[f"split{i}"] = predictions
        model_name = model.name
    return np.mean(scores), sem(scores), predictions_pd, model_name # the last model used - the same for all runs

def model_train(at_split):
    splits_path = f"data/CAT_LARGE"
    score, predictions, model = train(FLAGS.with_context_data,
                                      FLAGS.model_name,
                                      splits_path=splits_path,
                                      the_split_to_use=at_split)
    #model.save() todo: fix.
    return score, predictions

def main(argv):

    if FLAGS.create_random_splits>0:
        # Prepare the data for Monte Carlo k-fold Cross Validation
        print(f"Splitting the data randomly into {FLAGS.create_random_splits} splits")
        split_to_random_sets(splits=FLAGS.create_random_splits)
    elif FLAGS.create_balanced_datasets>0:
        # Down-sample the splits
        print ("Creating balanced versions")
        for i in range(FLAGS.repeat): # recall to set this to the correct value
            create_balanced_datasets(f"data/CAT_LARGE/{i}")
    elif FLAGS.repeat == 0:
        # Run at a single split
        score, predictions = model_train(FLAGS.at_split)
        print(f"{score}")
        pd.DataFrame(predictions).to_csv(f"{FLAGS.model_name}.predictions.csv")
    elif FLAGS.repeat > 0:
        # Perform Monte Carlo Cross Validation
        # INFO: Recall to set "repeat" to the correct folds number
        score, sem, predictions_pd, model_name = repeat_experiment()
        predictions_pd.to_csv(f"{FLAGS.experiment_version_name}/{model_name}.predictions.csv")
        print (f"{score} ± {sem}")

if __name__ == "__main__":
    app.run(main)
