import tensorflow.keras as keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM, Bidirectional
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import concatenate
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import backend as K
from bert import tokenization
from ACL2020.utils import InputExample, convert_examples_to_features
from sklearn.metrics import *
import pickle

BERT_MODEL_PATH = "https://tfhub.dev/google/bert_cased_L-12_H-768_A-12/1"

METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'),
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
]

def load_embeddings_index():
    embeddings_index = dict()
    with open('embeddings/glove.6B.100d.txt', 'r') as glove_in:
        for line in glove_in.readlines():
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index



class LSTM_CLF():
    def __init__(self, stacks=0, verbose=1, batch_size=128, n_epochs=100, max_length=512,
                 loss="binary_crossentropy", monitor_loss="val_loss", patience=3,
                 prefix="vanilla",
                 hidden_size=128,
                 word_embedding_size=200,
                 seed=42,
                 augmented_vocabulary = False,
                 no_sigmoid=False):
        tf.compat.v1.set_random_seed(seed)
        np.random.seed(seed)
        self.verbose = verbose
        self.augmented_vocabulary = augmented_vocabulary
        self.patience = patience
        self.batch_size = batch_size
        self.early = EarlyStopping(monitor="val_auc",
                                   mode="max",
                                   verbose=1,
                                   patience=self.patience,
                                   restore_best_weights=True
                                   )
        self.n_epochs = n_epochs
        self.no_sigmoid = no_sigmoid
        self.stacks=stacks
        self.max_length = max_length
        self.tokenizer = Tokenizer()
        self.loss = loss
        self.word_embedding_size = word_embedding_size
        self.hidden_size=hidden_size
        self.prefix = prefix
        self.monitor_loss = monitor_loss
        self.name = f'{prefix}-b{batch_size}.e{n_epochs}.len{max_length}.rnn'

    def load_embeddings(self, pretrained_dict):
        self.embedding_matrix = np.zeros((self.vocab_size + 2, 100))
        for word, index in self.tokenizer.word_index.items():
            embedding_vector = pretrained_dict.get(word)
            if embedding_vector is not None:
                self.embedding_matrix[index + 1] = embedding_vector

    def build(self, bias=0):
        inputs1 = Input(shape=(self.max_length,))
        stack = Embedding(self.vocab_size + 2, self.word_embedding_size, mask_zero=True)(inputs1)#, weights=[self.embedding_matrix], trainable=True)(inputs1)
        for i in range(self.stacks):
            stack = Bidirectional(LSTM(self.hidden_size, return_sequences=True))(stack)
        rnn = Bidirectional(LSTM(self.hidden_size, return_sequences=False))(stack)

        fnn = Dense(128, activation='tanh')(rnn)
        fnn = Dense(1, activation='sigmoid', bias_initializer=tf.keras.initializers.Constant(bias))(fnn)
        self.model = Model(inputs=inputs1, outputs=fnn)
        self.model.compile(loss=self.loss,
                           optimizer=keras.optimizers.Adam(),
                           metrics=METRICS)

    def model_show(self):
        print(self.model.summary())

    def text_process(self, texts):
        x1 = self.tokenizer.texts_to_sequences(texts.to_numpy())
        x1 = sequence.pad_sequences(x1, maxlen=self.max_length)  # padding
        return x1

    def fit(self, train, dev, pretrained_embeddings, class_weights={0: 1, 1: 1}):
        texts = train.text if not self.augmented_vocabulary else train.text + train.parent
        self.tokenizer.fit_on_texts(texts)
        self.vocab_size = len(self.tokenizer.word_index) + 1
        print('Vocabulary Size: %d' % self.vocab_size)
        X, VX = self.text_process(train.text), self.text_process(dev.text)
        Y, VY = train.label.to_numpy(), dev.label.to_numpy()
        self.load_embeddings(pretrained_embeddings)
        print (f"OLD-SCHOOL LOG: Building {self.name}...")
        pos = sum(Y)
        neg = len(Y)-pos
        bias = np.log(pos/neg)
        self.build(bias=bias)
        self.model_show()
        print (f"OLD-SCHOOL LOG: Training {self.name}...")
        self.history = self.model.fit(X, Y, validation_data=(VX, VY),
                                      epochs=self.n_epochs,
                                      batch_size=self.batch_size,
                                      verbose=self.verbose,
                                      callbacks=[self.early],
                                      class_weight=class_weights)

    def predict(self, test):
        predictions = self.model.predict(self.text_process(test.text))
        return predictions

    def save(self):
        self.model.save_weights(self.name + ".h5")
        del self.model
        with open(self.name+".arch", "wb") as out:
            out.write(pickle.dumps(self))
        self.model.load_weights(self.name + ".h5")

    def load(self):
        pickle.load(open(self.name+".arch"))
        self.build()
        self.model.load_weights(self.name+".h5")


class LSTM_IC1_CLF(LSTM_CLF):
    # RNN classification of the target text, with context representation concatenated.
    # The resulting representation of the target text is concatenated with the representation
    # of the parent text. The parent text representation comes from a single-level Bidirectional RNN.
    # The target text representation comes from a stacked LSTM.

    def __init__(self, prefix="IC1", **kwargs):
        super(LSTM_IC1_CLF, self).__init__(**kwargs)
        self.prefix = prefix

    def build(self, bias=0):
        target_input = Input(shape=(self.max_length,))
        stack = Embedding(self.vocab_size + 2, 200, mask_zero=True)(target_input)
        # stack = Embedding(self.vocab_size + 2, 200, mask_zero=True, weights=[self.embedding_matrix],
        #                  trainable=True)(target_input)
        for i in range(self.stacks):
            stack = LSTM(self.hidden_size, return_sequences=True)(stack)
        target_rnn = Bidirectional(LSTM(self.hidden_size, return_sequences=False))(stack)

        parent_input = Input(shape=(self.max_length,))
        parent_emb = Embedding(self.vocab_size + 2, 100, mask_zero=True)(parent_input)
        parent_rnn = Bidirectional(LSTM(64, return_sequences=False))(parent_emb)

        x = concatenate([target_rnn, parent_rnn])
        #x = keras.layers.Lambda(lambda embedding: K.l2_normalize(embedding, axis=1))(x)

        fnn = Dense(128, activation='tanh')(x)
        fnn = Dense(1, activation='sigmoid', bias_initializer=tf.keras.initializers.Constant(bias))(fnn)
        self.model = Model(inputs=[target_input, parent_input], outputs=fnn)
        self.model.compile(loss=self.loss,
                           optimizer=keras.optimizers.Adam(),
                           metrics=METRICS)

    def text_process(self, texts, parents):
        target_x = self.tokenizer.texts_to_sequences(texts.to_numpy())
        target_x = sequence.pad_sequences(target_x, maxlen=self.max_length)  # padding
        parent_x = self.tokenizer.texts_to_sequences(parents.to_numpy())
        parent_x = sequence.pad_sequences(parent_x, maxlen=self.max_length)  # padding
        return [target_x, parent_x]

    def fit(self, train, dev, pretrained_embeddings, class_weights={0: 1, 1: 1}):
        texts = train.text if not self.augmented_vocabulary else train.text + train.parent
        self.tokenizer.fit_on_texts(texts)
        self.vocab_size = len(self.tokenizer.word_index) + 1
        print('Vocabulary Size: %d' % self.vocab_size)
        X, VX = self.text_process(train.text, train.parent), self.text_process(dev.text, dev.parent)
        Y, VY = train.label.to_numpy(), dev.label.to_numpy()
        self.load_embeddings(pretrained_embeddings)
        pos = sum(Y)
        neg = len(Y)-pos
        bias = np.log(pos/neg)
        self.build(bias=bias)
        self.model_show()
        self.history = self.model.fit(X, Y,
                                      validation_data=(VX, VY),
                                      epochs=self.n_epochs,
                                      batch_size=self.batch_size,
                                      verbose=self.verbose,
                                      callbacks=[self.early],
                                      class_weight=class_weights)

    def predict(self, test):
        predictions = self.model.predict(self.text_process(test.text, test.parent))
        return predictions


class BERT(tf.keras.layers.Layer):
    """
    Extending the code from:
    https://towardsdatascience.com/fine-tuning-bert-with-keras-and-tf-module-ed24ea91cff2
    The layers to fine tuned are selected by name.
    """
    def __init__(
            self,
            n_fine_tune_top_layers=10,
            trainable=True,
            pooling="first",
            output_size=768,
            **kwargs,
    ):
        self.n_fine_tune_layers = n_fine_tune_top_layers
        self.trainable = trainable
        self.output_size = output_size
        self.pooling = pooling
        self.bert_path = BERT_MODEL_PATH
        if self.pooling not in ["first", "mean"]:
            raise NameError(f"Undefined pooling type (must be either first or mean, but is {self.pooling}")
        super(BERT, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'n_fine_tune_top_layers': self.n_fine_tune_top_layers,
            'trainable': self.trainable,
            'bert': self.bert,
            'pooling': self.pooling,
        })
        return config

    def build(self, input_shape):
        self.bert = hub.Module(
            self.bert_path, trainable=self.trainable, name=f"{self.name}_module"
        )

        # Remove unused layers
        trainable_vars = self.bert.variables
        if self.pooling == "first":
            trainable_vars = [var for var in trainable_vars if not "/cls/" in var.name]
            trainable_layers = ["pooler/dense"]

        elif self.pooling == "mean":
            trainable_vars = [
                var
                for var in trainable_vars
                if not "/cls/" in var.name and not "/pooler/" in var.name
            ]
            trainable_layers = []
        else:
            raise NameError(
                f"Undefined pooling type (must be either first or mean, but is {self.pooling}"
            )

        # Select how many layers to fine tune
        for i in range(self.n_fine_tune_layers):
            trainable_layers.append(f"encoder/layer_{str(11 - i)}")

        # Update trainable vars to contain only the specified layers
        trainable_vars = [
            var
            for var in trainable_vars
            if any([l in var.name for l in trainable_layers])
        ]

        # Add to trainable weights
        for var in trainable_vars:
            self._trainable_weights.append(var)

        for var in self.bert.variables:
            if var not in self._trainable_weights:
                self._non_trainable_weights.append(var)

        super(BERT, self).build(input_shape)

    def call(self, inputs):
        inputs = [K.cast(x, dtype="int32") for x in inputs]
        input_ids, input_mask, segment_ids = inputs
        bert_inputs = dict(
            input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids
        )
        if self.pooling == "first":
            pooled = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)[
                "pooled_output"
            ]
        elif self.pooling == "mean":
            result = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)[
                "sequence_output"
            ]

            mul_mask = lambda x, m: x * tf.expand_dims(m, axis=-1)
            masked_reduce_mean = lambda x, m: tf.reduce_sum(mul_mask(x, m), axis=1) / (
                    tf.reduce_sum(m, axis=1, keepdims=True) + 1e-10)
            input_mask = tf.cast(input_mask, tf.float32)
            pooled = masked_reduce_mean(result, input_mask)
        else:
            raise NameError(f"Undefined pooling type (must be either first or mean, but is {self.pooling}")

        return pooled

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_size)


class BERT_MLP():

    def __init__(self,
                 trainable_layers=3,
                 max_seq_length=128,
                 show_summary=False,
                 label_list=[0, 1],
                 patience=3,
                 seed=42,
                 epochs=100,
                 save_predictions=False,
                 batch_size=32,
                 DATA_COLUMN="text",
                 LABEL_COLUMN="label",
                 DATA2_COLUMN=None,
                 lr=2e-05,
                 session=None
                 ):
        self.session = session
        tf.compat.v1.set_random_seed(seed)
        np.random.seed(seed)
        self.name = f'{"OOC1" if not DATA2_COLUMN else "OOC2"}-b{batch_size}.e{epochs}.len{max_seq_length}.bert'
        self.tokenizer = self.create_tokenizer_from_hub_module()
        self.lr = lr
        self.batch_size = batch_size
        self.DATA_COLUMN=DATA_COLUMN
        self.DATA2_COLUMN=DATA2_COLUMN
        self.LABEL_COLUMN=LABEL_COLUMN
        self.trainable_layers = trainable_layers
        self.max_seq_length = max_seq_length
        self.show_summary = show_summary
        self.label_list = label_list
        self.patience=patience
        self.save_predictions = save_predictions
        self.epochs = epochs
        self.earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_auc',
                                                          patience=self.patience,
                                                          verbose=1,
                                                          restore_best_weights=True,
                                                          mode="max")

    def build(self, bias=0):
        in_id = tf.keras.layers.Input(shape=(self.max_seq_length,), name="input_ids")
        in_mask = tf.keras.layers.Input(shape=(self.max_seq_length,), name="input_masks")
        in_segment = tf.keras.layers.Input(shape=(self.max_seq_length,), name="segment_ids")
        bert_inputs = [in_id, in_mask, in_segment]
        bert_output = BERT(n_fine_tune_top_layers=self.trainable_layers)(bert_inputs)
        dense = tf.keras.layers.Dense(128, activation='tanh')(bert_output)
        pred = tf.keras.layers.Dense(1, activation='sigmoid', bias_initializer=tf.keras.initializers.Constant(bias))(dense)
        self.model = tf.keras.models.Model(inputs=bert_inputs, outputs=pred)
        self.model.compile(loss='binary_crossentropy',
                      optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
                      metrics=METRICS)
        if self.show_summary:
            self.model.summary()

    def get_features(self, features):
        input_ids, input_masks, segment_ids, labels = [], [], [], []
        for f in features:
            input_ids.append(f.input_ids)
            input_masks.append(f.input_mask)
            segment_ids.append(f.segment_ids)
            labels.append(f.label_id)
        return (np.array(input_ids), np.array(input_masks), np.array(segment_ids), np.array(labels).reshape(-1, 1),)

    def to_bert_input(self, dataset_pd):
        x_input = dataset_pd.apply(lambda x: InputExample(guid=None,
                                                          text_a=x[self.DATA_COLUMN],
                                                          text_b=x[self.DATA2_COLUMN] if self.DATA2_COLUMN else None,
                                                          label=x[self.LABEL_COLUMN]), axis=1)
        x_features = convert_examples_to_features(x_input,
                                                  self.label_list,
                                                  self.max_seq_length,
                                                  self.tokenizer)
        x_input_ids, x_input_masks, x_segment_ids, x_labels = self.get_features(x_features)
        return (x_input_ids, x_input_masks, x_segment_ids), x_labels

    def fit(self, train, dev, bert_weights=None, class_weights={0: 1, 1: 1}, pretrained_embeddings=None):
        train_input, train_labels = self.to_bert_input(train)
        dev_input, dev_labels = self.to_bert_input(dev)
        pos = sum(train_labels)
        neg = len(train_labels)-pos
        bias = np.log(pos/neg)
        print ("BIAS:", bias)
        self.build(bias=bias)
        if bert_weights is not None:
            self.model.load_weights(bert_weights)
        self.initialise_vars() # instantiation needs to be right before fitting
        self.model.fit(train_input,
                       train_labels,
                       validation_data=(dev_input, dev_labels),
                       epochs=self.epochs,
                       callbacks=[self.earlystop],
                       batch_size=self.batch_size,
                       class_weight=class_weights
                       )

    def predict(self, val_pd):
        #with self.session.as_default():
        val_input, val_labels = self.to_bert_input(val_pd)
        predictions = self.model.predict(val_input)
        score = roc_auc_score(val_labels, predictions)
        print('ROC AUC: {:.4f}'.format(score))
        print('Stopped epoch: ', self.earlystop.stopped_epoch)
        if self.save_predictions:
            self.save_evaluation_set(val_labels, predictions)
        return predictions

    def save_evaluation_set(self, gold, predictions):
        logtime = time.strftime('%Y%m%d-%H%M%S')
        pd.DataFrame({"gold":gold, "pred":predictions}).to_csv(f"{self.name}.{logtime}.evaluation.csv")

    def create_tokenizer_from_hub_module(self):
        bert_module = hub.Module(BERT_MODEL_PATH)
        tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
        vocab_file, do_lower_case = self.session.run([tokenization_info["vocab_file"],tokenization_info["do_lower_case"]])
        return tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

    def initialise_vars(self):
        self.session.run(tf.local_variables_initializer())
        self.session.run(tf.global_variables_initializer())
        self.session.run(tf.tables_initializer())
        K.set_session(self.session)

    def save(self):
        self.model.save(f"{self.name}.weights.h5")

    def model_show(self):
        print(self.model.summary())


class BERT_MLP_CA(BERT_MLP):

    def __init__(self, max_length=128, word_embedding_size=200, **kwargs):
        super(BERT_MLP_CA, self).__init__(**kwargs)
        self.name = f'{"CA"}-b{self.batch_size}.e{self.epochs}.len{self.max_seq_length}.bert'
        self.parent_tokenizer = Tokenizer()
        self.max_length = max_length
        self.word_embedding_size=word_embedding_size

    def load_embeddings(self, pretrained_dict):
        self.embedding_matrix = np.zeros((self.vocab_size + 2, 100))
        for word, index in self.parent_tokenizer.word_index.items():
            embedding_vector = pretrained_dict.get(word)
            if embedding_vector is not None:
                self.embedding_matrix[index + 1] = embedding_vector

    def build(self, bias=0):
        target_input = [Input(shape=(self.max_seq_length,), name="input_ids"),
                        Input(shape=(self.max_seq_length,), name="input_masks"),
                        Input(shape=(self.max_seq_length,), name="segment_ids")]
        target_output = BERT(n_fine_tune_top_layers=self.trainable_layers)(target_input)

        # add the parent
        parent_input = Input(shape=(self.max_length,), name="parent_input")
        parent_emb = Embedding(self.vocab_size + 2, self.word_embedding_size, mask_zero=True)(parent_input)
                               #weights=[self.embedding_matrix], trainable=True)(parent_input)
        # parent_emb = Embedding(self.vocab_size + 2, 100, mask_zero=True)(parent_input)
        parent_rnn = LSTM(128)(parent_emb)

        # concatenating and normalizing the two embeddings
        x = concatenate([target_output, parent_rnn])
        #x = keras.layers.Lambda(lambda embedding: K.l2_normalize(embedding, axis=1))(x)

        fnn = tf.keras.layers.Dense(128, activation='tanh')(x)
        fnn = Dense(1, activation='sigmoid', bias_initializer=tf.keras.initializers.Constant(bias))(fnn)
        #fnn = tf.keras.layers.Dense(1, activation='sigmoid')(fnn)
        self.model = tf.keras.models.Model(inputs=target_input+[parent_input], outputs=fnn)
        self.model.compile(loss='binary_crossentropy',
                      optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
                      metrics=METRICS)

    def fit(self, train, dev, pretrained_embeddings, bert_weights=None, batch_size=32, class_weights={0: 1, 1: 1}):
        self.parent_tokenizer.fit_on_texts(train.text)
        self.vocab_size = len(self.parent_tokenizer.word_index) + 1
        train_input, train_labels = self.to_bert_input(train)
        dev_input, dev_labels = self.to_bert_input(dev)
        parent_input = self.text_process(train.parent)
        parent_dev_input = self.text_process(dev.parent)
        self.load_embeddings(pretrained_embeddings)
        print (f"OLD-SCHOOL LOG: Building {self.name}...")
        pos = sum(train_labels)
        neg = len(train_labels)-pos
        bias = np.log(pos/neg)
        print ("BIAS:", bias)
        self.build(bias=bias)
        if bert_weights is not None:
            self.model.load_weights(bert_weights)
        self.model_show()
        print (f"OLD-SCHOOL LOG: Training {self.name}...")
        self.initialise_vars()
        self.history = self.model.fit(list(train_input)+[parent_input],
                                      train_labels,
                                      validation_data=(list(dev_input)+[parent_dev_input], dev_labels),
                                      epochs=self.epochs,
                                      callbacks=[self.earlystop],
                                      batch_size=batch_size,
                                      class_weight=class_weights
                                      )

    def text_process(self, texts):
        x = self.parent_tokenizer.texts_to_sequences(texts.to_numpy())
        x = sequence.pad_sequences(x, maxlen=self.max_length)  # padding
        return x

    def predict(self, val_pd):
        val_input, val_labels = self.to_bert_input(val_pd)
        parent_val_input = self.text_process(val_pd.parent)
        predictions = self.model.predict(list(val_input)+[parent_val_input])
        score = roc_auc_score(val_labels, predictions)
        print('ROC AUC: {:.4f}'.format(score))
        print('Stopped epoch: ', self.earlystop.stopped_epoch)
        if self.save_predictions:
            self.save_evaluation_set(val_labels, predictions)
        return predictions