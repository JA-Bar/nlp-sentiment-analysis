from pathlib import Path


import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences

from src.utils import load_from_pickle, interpret_results


def encode_text(input_text, data_path='data/', MAX_SEQUENCE_LENGTH=50):
    data_path = Path(data_path, 'rnn')
    tokenizer = load_from_pickle(data_path / "tokenizer.pkl")
    encoded_text = pad_sequences(
        tokenizer.texts_to_sequences(input_text),
        padding="post",
        maxlen=MAX_SEQUENCE_LENGTH,
    )
    return encoded_text


def get_bidirectional_rnn(pretrained=False,
                          rnn_path=None,
                          embedding_matrix=None,
                          vocab_size=309467,
                          EMBEDDING_DIM=200,
                          MAX_SEQUENCE_LENGTH=50):

    if pretrained:
        if rnn_path is None:
            raise ValueError("You must provide a path to pretrained models.")

        model_rnn_simple = tf.keras.models.load_model(rnn_path / "RNN_SIMPLE.h5")
    else:
        if embedding_matrix is None:
            raise AttributeError("To load a non-pretrained model, an embedding_matrix is needed.")

        model_rnn_simple = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size,
                                      EMBEDDING_DIM,
                                      weights=[embedding_matrix],
                                      input_length=MAX_SEQUENCE_LENGTH,
                                      trainable=False),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ])

        model_rnn_simple.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                                 optimizer=tf.keras.optimizers.Adam(1e-4),
                                 metrics=["accuracy"])

    return model_rnn_simple


def get_clstm(pretrained=False,
              rnn_path=None,
              vocab_size=309467,
              embedding_matrix=None,
              EMBEDDING_DIM=200,
              MAX_SEQUENCE_LENGTH=50):

    if pretrained:
        if rnn_path is None:
            raise ValueError("You must provide a path to pretrained models.")

        model_CLSTM = tf.keras.models.load_model(rnn_path / "CLSTM.h5")
    else:
        if embedding_matrix is None:
            raise AttributeError("To load a non-pretrained model, an embedding_matrix is needed.")

        model_CLSTM = tf.keras.Sequential([
                tf.keras.layers.Embedding(vocab_size,
                                          EMBEDDING_DIM,
                                          weights=[embedding_matrix],
                                          input_length=MAX_SEQUENCE_LENGTH,
                                          trainable=False),
                tf.keras.layers.SpatialDropout1D(0.2),
                tf.keras.layers.Conv1D(64, 5, activation="relu"),
                tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,
                                                                   dropout=0.2,
                                                                   recurrent_dropout=0.2)),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(1, activation="sigmoid"),
        ])

        model_CLSTM.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01),
                            loss="binary_crossentropy",
                            metrics=["accuracy"])

    return model_CLSTM


def get_gru_rnn(
    pretrained=False,
    rnn_path=None,
    vocab_size=309467,
    embedding_matrix=None,
    EMBEDDING_DIM=200,
    MAX_SEQUENCE_LENGTH=50):

    if pretrained:
        if rnn_path is None:
            raise ValueError("You must provide a path to pretrained models.")

        model_GRU = tf.keras.models.load_model(rnn_path/"GRU.h5")
    else:
        if embedding_matrix is None:
            raise AttributeError("To load a non-pretrained model, an embedding_matrix is needed.")

        model_GRU = tf.keras.Sequential([
                tf.keras.layers.Embedding(vocab_size,
                                          EMBEDDING_DIM,
                                          weights=[embedding_matrix],
                                          input_length=MAX_SEQUENCE_LENGTH,
                                          trainable=True),
                tf.keras.layers.GRU(units=16, return_sequences=True),
                tf.keras.layers.GRU(units=8, return_sequences=True),
                tf.keras.layers.GRU(units=4),
                tf.keras.layers.Dense(1, activation="sigmoid"),
        ])

        optimizer = tf.keras.optimizers.Adam(lr=1e-3)
        model_GRU.compile(loss="binary_crossentropy",
                          optimizer=optimizer,
                          metrics=["accuracy"])

    return model_GRU


def inference(sentences, data_path='data/', pretrained=True):
    input_text = encode_text(sentences, data_path=data_path)

    rnn_path = Path(data_path, 'rnn')

    if not pretrained:
        embedding_matrix = load_from_pickle(rnn_path / 'embedding_matrix.pkl')
    else:
        embedding_matrix = None

    biderectional_model = get_bidirectional_rnn(pretrained=True, rnn_path=rnn_path, embedding_matrix=embedding_matrix)
    biderectional_output = biderectional_model.predict(input_text)
    biderectional_output = interpret_results(biderectional_output, threshold=0)

    del biderectional_model

    clstm_model = get_clstm(pretrained=True, rnn_path=rnn_path, embedding_matrix=embedding_matrix)
    clstm_output = clstm_model.predict(input_text)
    clstm_output = interpret_results(clstm_output, threshold=0.5)

    del clstm_model

    gru_model = get_gru_rnn(pretrained=True, rnn_path=rnn_path, embedding_matrix=embedding_matrix)
    gru_ouput = gru_model.predict(input_text)
    gru_ouput = interpret_results(gru_ouput, threshold=0.5)

    del gru_model

    return {
        "bidirectional": biderectional_output,
        "clstm": clstm_output,
        "gru": gru_ouput,
    }
