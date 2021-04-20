import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences

from src.utils import load_from_pickle


def encode_text(input_text, MAX_SEQUENCE_LENGTH=50):
    tokenizer = load_from_pickle("data/tokenizer.pkl")
    encoded_text = pad_sequences(
        tokenizer.texts_to_sequences(input_text),
        padding="post",
        maxlen=MAX_SEQUENCE_LENGTH,
    )
    return encoded_text


def get_bidirectional_rnn(
    pretrained=False,
    embedding_matrix=None,
    vocab_size=309467,
    EMBEDDING_DIM=200,
    MAX_SEQUENCE_LENGTH=50,
):
    if pretrained:
        model_rnn_simple = tf.keras.models.load_model("data/RNN_SIMPLE.h5")
    else:
        if embedding_matrix is None:
            raise AttributeError(
                "To load a non-pretrained model, an embedding_matrix is needed."
            )

        model_rnn_simple = tf.keras.Sequential(
            [
                tf.keras.layers.Embedding(
                    vocab_size,
                    EMBEDDING_DIM,
                    weights=[embedding_matrix],
                    input_length=MAX_SEQUENCE_LENGTH,
                    trainable=False,
                ),
                tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )

        model_rnn_simple.compile(
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            optimizer=tf.keras.optimizers.Adam(1e-4),
            metrics=["accuracy"],
        )

    return model_rnn_simple


def get_clstm(
    pretrained=False,
    vocab_size=309467,
    embedding_matrix=None,
    EMBEDDING_DIM=200,
    MAX_SEQUENCE_LENGTH=50,
):
    if pretrained:
        model_CLSTM = tf.keras.models.load_model("data/CLSTM.h5")
    else:
        if embedding_matrix is None:
            raise AttributeError(
                "To load a non-pretrained model, an embedding_matrix is needed."
            )

        model_CLSTM = tf.keras.Sequential(
            [
                tf.keras.layers.Embedding(
                    vocab_size,
                    EMBEDDING_DIM,
                    weights=[embedding_matrix],
                    input_length=MAX_SEQUENCE_LENGTH,
                    trainable=False,
                ),
                tf.keras.layers.SpatialDropout1D(0.2),
                tf.keras.layers.Conv1D(64, 5, activation="relu"),
                tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2)
                ),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )

        model_CLSTM.compile(
            optimizer=tf.keras.optimizers.Adam(lr=0.01),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

    return model_CLSTM


def get_gru_rnn(
    pretrained=False,
    vocab_size=309467,
    embedding_matrix=None,
    EMBEDDING_DIM=200,
    MAX_SEQUENCE_LENGTH=50,
):

    if pretrained:
        model_GRU = tf.keras.models.load_model("data/GRU.h5")
    else:
        if embedding_matrix is None:
            raise AttributeError(
                "To load a non-pretrained model, an embedding_matrix is needed."
            )

    model_GRU = tf.keras.Sequential(
        [
            tf.keras.layers.Embedding(
                vocab_size,
                EMBEDDING_DIM,
                weights=[embedding_matrix],
                input_length=MAX_SEQUENCE_LENGTH,
                trainable=True,
            ),
            tf.keras.layers.GRU(units=16, return_sequences=True),
            tf.keras.layers.GRU(units=8, return_sequences=True),
            tf.keras.layers.GRU(units=4),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    optimizer = tf.keras.optimizers.Adam(lr=1e-3)
    model_GRU.compile(
        loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"]
    )


def inference(sentences):
    input_text = encode_text(sentences)

    embedding_matrix = load_from_pickle("data/embedding_matrix.pkl")

    biderectional = get_bidirectional_rnn(
        pretrained=True, embedding_matrix=embedding_matrix
    )
    clstm = get_clstm(pretrained=True, embedding_matrix=embedding_matrix)
    gru = get_gru_rnn(pretrained=True, embedding_matrix=embedding_matrix)

    biderectional_output = biderectional.predict(input_text)
    clstm_output = clstm.predict(input_text)
    gru_ouput = gru.predict(input_text)

    return {
        "bidirectional": biderectional_output,
        "clstm": clstm_output,
        "gru": gru_ouput,
    }
