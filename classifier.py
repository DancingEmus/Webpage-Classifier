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


def tokenizer(root_folder):
    oov_tok = '<OOV>'
    vocab_size = 5000
    max_length = 0
    padding_type = 'post'
    embedding_dim = 64
    val_ratio = .60  # Ratio of validation data vs test data

    dirs = [dirs for root, dirs, files in os.walk(root_folder, topdown=True)]
    dirs[0].remove("links")
    categories = []
    word_sequences = []
    categories_index = {}
    [categories_index.update({dir: i}) for i, dir in enumerate(dirs[0])]

    # Us os.walk to categorize and retrieve dataset data from scraper.py.
    for dir1 in dirs[0]:
        dir_path = os.path.join(root_folder, dir1)
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
    random.shuffle(temp)
    word_sequences, categories = zip(*temp)

    # Index to split training data by validation data.
    num_validation = math.floor(len(word_sequences) * val_ratio)
    num_train = len(word_sequences) - num_validation

    # Get the average size of the batches and make that the min length.
    sum_seq = 0
    for seq in word_sequences:
        sum_seq += len(seq)
    max_length = math.floor(sum_seq / len(word_sequences))

    # Fit tokenizer vocab on word_sequences from json files.
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(word_sequences)
    word_index = tokenizer.word_index

    # Tokenized training sequences.
    train_sequences = tokenizer.texts_to_sequences(word_sequences[:num_train])
    train_padded = pad_sequences(train_sequences, padding=padding_type, maxlen=max_length)

    # Tokenized validation sequences.
    validation_sequences = tokenizer.texts_to_sequences(word_sequences[num_train:])
    validation_padded = pad_sequences(validation_sequences, padding=padding_type, maxlen=max_length)

    # Train label seq as dictionary values.
    train_label_seq = [categories_index.get(seq) for seq in categories[:num_train]]
    train_label_seq = np.array([train_label_seq]).transpose()

    # Validation label seq as dictionary values.
    validation_label_seq = [categories_index.get(seq) for seq in categories[num_train:]]
    validation_label_seq = np.array([validation_label_seq]).transpose()

    return vocab_size, embedding_dim, max_length, train_padded, train_label_seq, validation_padded, \
        validation_label_seq, word_index, categories_index


def train_model(vocab_size, embedding_dim, max_length, train_padded, train_label_seq, validation_padded,
                validation_label_seq):
    # Keras embedding layer initialization
    embedding_layer = tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length)

    # NN Model with embedding layer.
    model = tf.keras.Sequential([
        embedding_layer,
        tf.keras.layers.GlobalMaxPool1D(),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(4, activation='softmax')
    ])

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    num_epochs = 30

    history = model.fit(x=train_padded, y=train_label_seq, epochs=num_epochs,
                        validation_data=(validation_padded, validation_label_seq), verbose=2)

    return history, model


def main():
    vocab_size, embedding_dim, max_length, train_padded, train_label_seq, validation_padded, \
        validation_label_seq, word_index, categories_index = tokenizer("dataset")
    history, model = train_model(vocab_size, embedding_dim, max_length, train_padded, train_label_seq,
                                 validation_padded, validation_label_seq)

    plot_graphs(history, "accuracy")
    plot_graphs(history, "loss")

    results = model.predict(train_padded[0])
    results = np.argmax(results, axis=1)
    result = np.argmax(np.bincount(results))

    # Reverse of categories index dict.
    index_categories = {v: k for k, v in categories_index.items()}
    # Reverse Word Index to introduce novel sequences.
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

    prediction = index_categories[result]
    actual = index_categories[train_label_seq[0][0]]
    print(f'prediction: {prediction} || actual: {actual}')

    if prediction != actual:
        print(decode_sentence(train_padded[0], reverse_word_index))
    e = model.layers[0]
    weights = e.get_weights()[0]
    print(weights.shape)  # shape: (vocab_size, embedding_dim)

    # Output the vector file and meta data file to tab separated value files.
    out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
    out_m = io.open('meta.tsv', 'w', encoding='utf-8')

    for word_num in range(1, vocab_size):
        word = reverse_word_index[word_num]
        embeddings = weights[word_num]
        out_m.write(word + "\n")
        out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
    out_v.close()
    out_m.close()


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.savefig(f"{string}.png")


def decode_sentence(text, reverse_word_index):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


main()
