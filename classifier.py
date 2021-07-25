import json
import os
import io
import math
import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


class Classifier:
    oov_tok = '<OOV>'
    vocab_size = 8000
    max_length = 0
    padding_type = 'post'
    embedding_dim = 64
    num_epochs = 25
    val_ratio = .25  # Ratio of validation data vs test data
    root_folder = 'dataset2'
    train_padded, train_label_seq, validation_padded, validation_label_seq = "", "", "", ""
    categories_index = {}
    word_index = []

    def tokenizer(self):
        dirs = [dirs for root, dirs, files in os.walk(self.root_folder, topdown=True)]
        dirs[0].remove("links")
        categories = []
        word_sequences = []

        [self.categories_index.update({dir1: i}) for i, dir1 in enumerate(dirs[0])]

        # Us os.walk to categorize and retrieve dataset data from scraper.py.
        for dir1 in dirs[0]:
            dir_path = os.path.join(self.root_folder, dir1)
            files = [files for root, dirs, files in os.walk(dir_path, topdown=True)]
            for file in files[0]:
                dir_path
                file_path = os.path.join(dir_path, file)
                with open(file_path) as f:
                    data = json.load(f)
                    word_sequences.append(data)
                    categories.append(dir1)

        # Shuffle the word sequences and categories so that we can grab random subsets of each category.
        temp = list(zip(word_sequences, categories))
        random.seed(3)
        random.shuffle(temp)
        word_sequences, categories = zip(*temp)

        # Index to split training data by validation data.
        num_validation = math.floor(len(word_sequences) * self.val_ratio)
        num_train = len(word_sequences) - num_validation

        # Get the average size of the batches and make that the min length.
        sum_seq = 0
        for seq in word_sequences:
            sum_seq += len(seq)
        self.max_length = math.floor(sum_seq / len(word_sequences))
        # max_length = 1000

        # Fit tokenizer vocab on word_sequences from json files.
        tokenizer = Tokenizer(num_words=self.vocab_size, oov_token=self.oov_tok)
        tokenizer.fit_on_texts(word_sequences)
        self.word_index = tokenizer.word_index

        # Tokenized training sequences.
        self.train_sequences = tokenizer.texts_to_sequences(word_sequences[:num_train])
        self.train_padded = pad_sequences(self.train_sequences, padding=self.padding_type, maxlen=self.max_length)

        # Tokenized validation sequences.
        self.validation_sequences = tokenizer.texts_to_sequences(word_sequences[num_train:])
        self.validation_padded = pad_sequences(self.validation_sequences, padding=self.padding_type, maxlen=self.max_length)

        # Train label seq as dictionary values.
        self.train_label_seq = [self.categories_index.get(seq) for seq in categories[:num_train]]
        self.train_label_seq = np.array([self.train_label_seq]).transpose()

        # Validation label seq as dictionary values.
        self.validation_label_seq = [self.categories_index.get(seq) for seq in categories[num_train:]]
        self.validation_label_seq = np.array([self.validation_label_seq]).transpose()

    def train_model(self):
        # Keras embedding layer initialization
        embedding_layer = tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim, input_length=self.max_length)

        # NN Model with embedding layer.
        model = tf.keras.Sequential([
            embedding_layer,
            tf.keras.layers.GlobalMaxPool1D(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(4, activation='softmax')
        ])

        model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.summary()

        history = model.fit(x=self.train_padded, y=self.train_label_seq, epochs=self.num_epochs,
                            validation_data=(self.validation_padded, self.validation_label_seq), verbose=2)

        return history, model

    def __init__(self):
        # Call the Tokenizer function to tokenize the word_sequences and split into training data/validation data sets.
        self.tokenizer()

        # Train model using out tokenized sequences and labels.
        history, model = self.train_model()

        # Plot the acc and loss of our training and validation data.
        self.plot_graphs(history, "accuracy")
        self.plot_graphs(history, "loss")

        # Reverse Word Index to introduce novel sequences.
        reverse_word_index = dict([(value, key) for (key, value) in self.word_index.items()])
        for i in range(self.validation_padded.shape[0]):
            results = model.predict(np.reshape(self.validation_padded[i], (1, self.validation_padded[i].shape[0])))
            result = np.argmax(results, axis=1)
            confidence = np.max(results)

            # Reverse of categories index dict.
            index_categories = {v: k for k, v in self.categories_index.items()}

            prediction = index_categories[result[0]]
            actual = index_categories[self.validation_label_seq[i][0]]
            print(f'prediction: {prediction} || actual: {actual} || (confidence={math.floor(confidence * 100)}%)')

            if prediction != actual:
                print(self.decode_sentence(self.validation_padded[i], reverse_word_index))
        e = model.layers[0]
        weights = e.get_weights()[0]
        print(weights.shape)  # shape: (vocab_size, embedding_dim)

        # Output the vector file and meta data file to tab separated value files.
        out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
        out_m = io.open('meta.tsv', 'w', encoding='utf-8')

        for word_num in range(1, self.vocab_size):
            word = reverse_word_index[word_num]
            embeddings = weights[word_num]
            out_m.write(word + "\n")
            out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
        out_v.close()
        out_m.close()

    def plot_graphs(self, history, string):
        plt.plot(history.history[string])
        plt.plot(history.history['val_' + string])
        plt.xlabel("Epochs")
        plt.ylabel(string)
        plt.legend([string, 'val_' + string])
        plt.savefig(f"{string}.png")

    def decode_sentence(self, text, reverse_word_index):
        return ' '.join([reverse_word_index.get(i, '?') for i in text])

Classifier()