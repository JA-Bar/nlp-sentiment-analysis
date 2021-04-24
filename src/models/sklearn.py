from pathlib import Path

import numpy as np
from nltk.tokenize import TweetTokenizer
from pyphonetics import Lein

from src.utils import load_from_pickle


def df_to_embedded_df(text, word_embedded, embeddings_phonetics):
    phonetic_encoder = Lein()
    tknzr = TweetTokenizer()
    text_as_list = tknzr.tokenize(text)

    embedded_text = []
    for word in text_as_list:
        if word in word_embedded:
            embedded_text.append(word_embedded[word])
        elif phonetic_encoder.phonetics(word) in embeddings_phonetics:
            embedded_text.append(embeddings_phonetics[phonetic_encoder.phonetics(word)])
        if len(embedded_text) == 0:
            embedded_text = np.nan
    return embedded_text


def inference(lista, data_path='data/'):
    data_path = Path(data_path, 'sklearn')

    word_embedded = load_from_pickle(data_path / "word_embedded")
    embeddings_phonetics = load_from_pickle(data_path / "embeddings_phonetics")
    prediction_dictionary = {}
    text_embbeded = [df_to_embedded_df(element, word_embedded, embeddings_phonetics) for element in lista]
    embbeded_mean = [np.mean(element, axis=0).reshape(1, -1) for element in text_embbeded]

    BernoulliNB_classifier = load_from_pickle(data_path / "BernoulliNB_classifier_embedded")
    prediction = [BernoulliNB_classifier.predict(text) for text in embbeded_mean]
    prediction_dictionary["BernoulliNB"] = prediction
    del BernoulliNB_classifier

    SGDClassifier_classifier = load_from_pickle(data_path / "SGDClassifier_classifier_embbeded")
    prediction = [SGDClassifier_classifier.predict(text) for text in embbeded_mean]
    prediction_dictionary["SGDClassifier"] = prediction
    del SGDClassifier_classifier

    LogisticRegression_classifier = load_from_pickle(data_path / "LogisticRegression_classifier_embbeded")
    prediction = [LogisticRegression_classifier.predict(text) for text in embbeded_mean]
    prediction_dictionary["LogisticRegression"] = prediction
    del LogisticRegression_classifier

    RandomForest_classifier = load_from_pickle(data_path / "SGDClassifier_classifier_embbeded")
    prediction = [RandomForest_classifier.predict(text) for text in embbeded_mean]
    prediction_dictionary["RandomForest"] = prediction
    del RandomForest_classifier

    del word_embedded
    del embeddings_phonetics

    return prediction_dictionary

