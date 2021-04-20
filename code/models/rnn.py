import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences

from code.utils import load_from_pickle


def encode_text(input_text, MAX_SEQUENCE_LENGTH=50):
    tokenizer = load_from_pickle("data/tokenizer.pkl")
    encoded_text = pad_sequences(tokenizer.texts_to_sequences(input_text),
                                 padding='post',
                                 maxlen=MAX_SEQUENCE_LENGTH)
    return encoded_text


def get_bidirectional_rnn(pretrained=False,
                          vocab_size=309467,
                          EMBEDDING_DIM=200,
                          MAX_SEQUENCE_LENGTH=50):
    if pretrained:
        model_rnn_simple = tf.keras.models.load_model("data/RNN_SIMPLE.h5")
    else:
        embedding_matrix = load_from_pickle("data/embedding_matrix.pkl")

        model_rnn_simple = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size, EMBEDDING_DIM,
                                      weights=[embedding_matrix],
                                      input_length=MAX_SEQUENCE_LENGTH,
                                      trainable=False),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        model_rnn_simple.compile(
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            optimizer=tf.keras.optimizers.Adam(1e-4),
            metrics=['accuracy']
        )

    return model_rnn_simple


def inference(sentences):
    input_text = encode_text(sentences)

    biderectional_rnn = get_bidirectional_rnn(pretrained=True)
    biderectional_rnn_output = biderectional_rnn.predict(input_text)

    return {
        'bidirectional_rnn': biderectional_rnn_output,
    }

