# Keras
import keras
from keras.utils import pad_sequences
from keras.preprocessing.text import Tokenizer, tokenizer_from_json

import json
import os

# Hyperparameter tuning:
import keras_tuner
from keras_tuner import HyperModel

# Keras model
import tensorflow as tf

from keras.models import Sequential
from keras import layers

# Data processing
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

# plotting
import matplotlib.pyplot as plt

plt.style.use('ggplot')

# relative dataset paths
datasets = {'yelp': 'datasets/yelp_labelled.txt',
            'amazon': 'datasets/amazon_cells_labelled.txt',
            'imdb': 'datasets/imdb_labelled.txt'}


class SentimentHyperModel(keras_tuner.HyperModel):

    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size  # total vocabulary size

    def build(self, hp):
        # building a sequential model while tuning hyperparameters
        model = Sequential()

        # word embedding layer
        model.add(
            layers.Embedding(input_dim=self.vocab_size, output_dim=hp.Int("embedding_dim", min_value=8, max_value=16)))

        # 1d pooling layer - squish the data into one dimension using average on the first axis
        model.add(layers.GlobalAvgPool1D())

        # two inner hidden layers
        model.add(layers.Dense(units=hp.Int("units1", min_value=16, max_value=128, step=16), activation="relu"))

        model.add(layers.Dense(units=hp.Int("units2", min_value=16, max_value=128, step=16), activation="relu"))

        # output layer
        model.add(layers.Dense(1, activation="sigmoid"))

        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics='accuracy', )

        return model

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            verbose=False,

            # tuning number of sentences in a batch
            batch_size=hp.Int("batch_size", min_value=5, max_value=100, step=5),

            # Tune whether to shuffle the data in each batch.
            shuffle=hp.Boolean("shuffle"),
            **kwargs,
        )


# get the ready dataset with all sentences
def prepare_data():
    df_list = []

    # concatenating datasets into one dataframe

    for source, filepath in datasets.items():
        df = pd.read_csv(filepath, names=['sentence', 'label'], sep='\t')
        df['source'] = source  # Add another column filled with the source name
        df_list.append(df)

    df = pd.concat(df_list)

    return df


# Plot history of model training
def plot_history(history):
    # Accuracy
    plt.subplot(1, 2, 1)

    plt.plot(history.history['accuracy'], linewidth=3)
    plt.plot(history.history['val_accuracy'], linewidth=3)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')

    # Loss
    plt.subplot(1, 2, 2)
    loss_values = history.history['loss']
    epochs = range(1, len(loss_values) + 1)

    plt.title('model loss')
    plt.plot(epochs, loss_values, label='Training Loss', linewidth=3)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # showing the graph with two subplots
    plt.show()


# wrapper over the keras model
class SentimentVerboseModel():
    maxlen = 450  # maximum length of a single review (all sentences are chopped to this length)

    def __init__(self):
        self.model = None  # trained keras model
        self.df = prepare_data()  # getting the ready data
        self.tokenizer = Tokenizer(num_words=len(self.df['sentence']))  # tokenizer to assign numbers to words

    def is_trained(self):
        return self.model is not None

    def predict(self, texts):
        if self.model is None:
            raise RuntimeError("Model is empty")

        pred_arr = self.model.predict(self.texts_to_sequences(texts))  # array of predicted singleton vectors
        return [round(val[0]) for val in pred_arr]

    def texts_to_sequences(self, texts):
        return pad_sequences(self.tokenizer.texts_to_sequences(texts), padding='post',
                                        maxlen=SentimentVerboseModel.maxlen)

    def save(self, path):
        if self.model is None:
            raise RuntimeError("Model is empty")

        # saving the model
        self.model.save(path)

        # saving the tokenizer
        tokenizer_json = self.tokenizer.to_json()
        with open(os.path.join(path, 'tokenizer.json', ), 'w', encoding='utf-8') as f:
            f.write(json.dumps(tokenizer_json, ensure_ascii=False))

    def load_model(self, path):
        # loading the model
        self.model = keras.models.load_model(path)

        # loading the tokenizer
        with open(os.path.join(path, 'tokenizer.json'), 'r') as f:
            data = json.load(f)  # json data from file
            self.tokenizer = tokenizer_from_json(data)  # creating tokenizer object from the data

    def verbose_train(self):
        print("DataFrame:\n", self.df.head(), "\n")

        sentences = self.df['sentence'].values
        y_column = self.df['label'].values

        print("Sentences: \n", sentences, "\n")

        print("Labels: \n", y_column, "\n")

        # Splitting the data into test and train sets
        X_train, X_test, y_train, y_test = train_test_split(sentences, y_column, test_size=0.2,
                                                            random_state=1000)
        # assigning numbers to words
        self.tokenizer.fit_on_texts(sentences)

        # Tokenize the input matrices and add padding characters up to self.maxlen symbols
        X_train = self.texts_to_sequences(X_train)
        X_test = self.texts_to_sequences(X_test)
        X_all = self.texts_to_sequences(sentences)

        vocab_size = len(self.tokenizer.word_index)  # total words in the vocabulary

        # Hyperparameter tuning
        hypermodel = SentimentHyperModel(len(sentences))
        hp = keras_tuner.HyperParameters()

        # Defining tuner and starting search
        tuner = keras_tuner.RandomSearch(
            hypermodel=hypermodel,
            objective="val_accuracy",
            max_trials=10,
            executions_per_trial=1,
            overwrite=True,
        )

        maxepochs = 50
        tuner.search(X_train, y_train, epochs=maxepochs,
                     callbacks=[tf.keras.callbacks.EarlyStopping('accuracy', patience=6)],
                     validation_data=(X_test, y_test))

        # Query the best model
        model = tuner.get_best_models(num_models=1)[0]
        best_hp = tuner.get_best_hyperparameters()[0]
        print(model.summary())

        # Print the model evaluation details
        loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
        print("Training Accuracy: {:.4f}".format(accuracy))

        loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
        print("Testing Accuracy:  {:.4f}".format(accuracy))

        # Retrain on train data to display learning plot
        print("Acquiring the best model and plotting it's accuracy & lost info")
        model = hypermodel.build(best_hp)

        history = hypermodel.fit(best_hp, model, X_train, y_train, validation_data=(X_test, y_test),
                                 epochs=maxepochs,
                                 callbacks=[tf.keras.callbacks.EarlyStopping('accuracy', patience=6)])
        plot_history(history)

        # Retrain the model with the best hyperparameters on the entire dataset
        print("Retraining the best model on the entire dataset")
        model = hypermodel.build(best_hp)
        hypermodel.fit(best_hp, model, X_all, y_column, epochs=maxepochs,
                       callbacks=[tf.keras.callbacks.EarlyStopping('accuracy', patience=6)])

        self.model = model