import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import pandas as pd

import preprocessor as p

import pickle
import contractions

import matplotlib.pyplot as plt

import re
from nltk.tokenize import TweetTokenizer


nltk.download('stopwords')
nltk.download('wordnet')


def save_pickle(name_to_save, document):
    name_to_save = open(f"drive/My Drive/Colab Notebooks/sentiment/{name_to_save}.pkl", "wb")
    pickle.dump(document, name_to_save)
    name_to_save.close()


def label_decoder(label, lab_to_sentiment):
    return lab_to_sentiment[label]


def df_cleaning(path_cv='drive/My Drive/Colab Notebooks/training.1600000.processed.noemoticon.csv'):
    df = pd.read_csv(path_cv, encoding='latin-1', header=None)
    # Label the columns of our dataset
    df.columns = ['sentiment', 'id', 'date', 'query', 'user_id', 'text']
    # Drop the columns that we won't need
    df = df.drop(['id', 'date', 'query', 'user_id'], axis=1)
    # For this example, we are only keeping 3% of the data but ideally
    # we will train with the whole dataset if we had the computation power
    df = df.sample(frac=1, random_state=1)
    df = df.drop_duplicates(subset=["text"])
    # Dictionary to replace the numbers to a Binary classification
    lab_to_sentiment = {0: 0, 4: 1}
    df.sentiment = df.sentiment.apply(lambda x: label_decoder(x))
    return df
    

def visualize(df):
    # Print the final data
    df.head()
    val_count = df.sentiment.value_counts()
    plt.figure(figsize=(8, 4))
    plt.bar(val_count.index, val_count.values)
    plt.title("Sentiment Data Distribution")


def not_all_stop_words(stop_words):
    stop_words.remove("not")
    stop_words.remove("no")
    stop_words.remove("but")
    stop_words.remove("nor")
    lista = ["'t", "nt"]

    for word in stop_words:
        if any(element in word for element in lista):
            stop_words.remove(word)
    return stop_words

lemmatizer = WordNetLemmatizer()


def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None
    
def preprocess_each_text(text, stop_words, tknzr):
        text = p.clean(text)
        # We get rid of the links on the tweets + lowercase + blank spaces at the end and beginning
        text = normalize(text)
        tokens = []
        # we split into words our tweet
        words = tknzr.tokenize(text)
        nltk_tagged = nltk.pos_tag(words)  
        #tuple of (token, wordnet_tag)
        wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
        lemmatized_sentence = []

        for word, tag in wordnet_tagged:
            if tag is None:
                if word not in stop_words:
                    #if there is no available tag, append the token as is
                    all_words.append(word)
                    lemmatized_sentence.append(word)
            else:
                if word not in stop_words:
                    #else use the tag to lemmatize the token
                    lemma = lemmatizer.lemmatize(word, tag)
                    lemmatized_sentence.append(lemma)
        return " ".join(lemmatized_sentence)


def preprocess(text):
    # We load stop words from nltk corpus
    stop_words = stopwords.words('english')
    #stemmer = SnowballStemmer('english')
    # We created a regex that will help us clean all of the links that are
    # attached in our data
    stop_words = not_all_stop_words(stop_words)
    # For this example, we found a TweetTokenizer which suppose to do a better job
    # at tokenize the words on a Tweet
    tknzr = TweetTokenizer()
    text = preprocess_each_text(text, stop_words, tknzr)
    
    return text

    # We define a function that will help us preprocess every row in our data
    


def no_acent(sentence):
    replacements = (
        ("á", "a"),
        ("é", "e"),
        ("í", "i"),
        ("ó", "o"),
        ("ú", "u"),
    )
    for acento, sin_acento in replacements:
        sentence = sentence.replace(acento, sin_acento).replace(acento.upper(), sin_acento.upper())
    return sentence


def normalize(text):
    # contractions
    text = text.lower()
    text = no_acent(text)
    text = contractions.fix(text)
    text = text.replace('\n',' . ')

    # Clean characters
    text = re.sub(r'[^A-z0-9!?.,\':&]', ' ', text)
    text =  re.sub(r'[^a-zA-z.,!?/:;\"\'\s]', '', text)
    text = text.replace('_', ' ')

    # deal with special mark
    text = text.replace('&', ' and ')
    text = re.sub(r':\'\(', ' , ', text)
    text = re.sub(r'[([)\]]', ' ', text)
    text = re.sub(r':[A-Z]', ' ', text)
    text = re.sub(r':','', text)
    text = re.sub(r'\*', '', text)
    text = re.sub(r'[/\\]', ' ', text)
    text = re.sub(r', \' \.', ' . ', text)
    text = re.sub(r'&+', ' and ', text)
    text = re.sub(r'(,\s*\.)+', ' . ', text)
    text = re.sub(r'(\.\s*,)+', ' . ', text)

    # deal with repeating marks
    text = re.sub(r'(,\s*)+', ' , ', text)

    # remove all single characters
    text =  re.sub(r"\b[a-zA-Z]\b", "", text)

    # remove numbers in words
    text = re.sub(r'[^a-zA-z.,!?/:;\"\'\s]', '', text)

    # Remove extra
    text = re.sub(r'(([^\w\s])\2+)',"",text)

    # remove www. ___ .com
    text = re.sub(r'(www|http)\S+',"",text)

    return text


if __name__ == '__main__':
    df.text = df.text.apply(lambda x: preprocess(x))
    save_pickle("final_df_preprocessed", df)

    df.head()

    train_data = df[:round(len(df)*.8)]
    test_data = df[round(len(df)*.8):]

    print(len(train_data))
    print(len(test_data))

    train_data.head()

    save_pickle("train_preproccesed_data", train_data)

    save_pickle("test_preproccesed_data", test_data)

