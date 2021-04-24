import argparse

from src import models
from src.models import ensemble, voice_recognition
from src.apis import twitter_user, reddit


def format_results(sentences, results):
    classes = {0: 'Negative', 1: 'Positive'}

    formatted_results = []
    for sent, res in zip(sentences, results):
        formatted_results.append((sent, classes[res]))

    return formatted_results


def predict(sentences, data_path='data/'):
    models.bert.prepare_model(data_path)

    results, inferences = ensemble.inference(sentences, data_path=data_path, return_inferences=True)
    results = format_results(sentences, results)

    print("The individual inferences by the models were: ", inferences)
    print("\n\n\n")
    print("The final result by the ensemble is: ", results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', action='store_true')
    parser.add_argument('--input', default="")
    parser.add_argument('--voice', action='store_true')
    parser.add_argument('--twitter', default="")
    parser.add_argument('--reddit', default="")
    parser.add_argument('--data_path', default="data/")

    args = parser.parse_args()
    sentences = ""

    if args.demo:
        sentences = [
            "I don't like the way this day is going.",
            "The movie was quite bad to be honest with you.",
            "I feel like I'm killing it",
            "No more content like this please.",
            "This piece of music is really enjoyable."
        ]
    elif args.input:
        sentences = args.input.split("&&")
    elif args.voice:
        sentences = voice_recognition.audio_to_string()
    elif args.twitter:
        sentences = twitter_user.tweets_from_user(args.twitter)
    elif args.reddit:
        sentences = reddit.comments_from_user(args.reddit)

    print('The input sentences are:', sentences)

    predict(sentences, args.data_path)

