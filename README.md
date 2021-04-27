# NLP Integration Project: Sentiment Analysis

Group integration project of the NLP course. The purpose of this project
is to implement different models and preprocessing techniques to perform
sentiment analysis on a series of short texts.

## Description

We implemented three model types:
- Classical ML models: Implemented using scikit-learn. The current models are 
  BernoulliNB, SGDClassifier, LogisticRegression, and RandomForest.
- RNNs: Built and trained using keras, they are a Bidirectional RNN, CLSTM, and GRU.
- BERT: A BERT model trained using Keras.

The inferences of all the models are then combined in an ensemble to produce
a final prediction. The predictions of all the models are weighed equally.

## Requirements

- Install PortAudio, if using a debian based linux distribution use the command:
  `sudo apt-get install libportaudio2`
- Optionally create and source the python virtual environment of your choice.
- Install pytorch>=1.7 using [the official page][1] according to your system.
- Run `pip install -r requirements.txt`


## Interface

The interface with this ensemble is done via the `src.inference` script. The colab notebook
`NLP-showcase.ipynb` demonstrates the different modes of operation.

To use this script simply run it as a module and provide flags, which act as
an input specifier to the ensemble.

`python3 -m src.inference --demo`

The possible flags are:
- --demo: Predict the sentiment of a fixed, predefined set of sentences.
- --input: Predict over a user-given sentences. The sentences should be given 
  as a string argument, where the set of characters '&&' e.g. 
  'this is sentence one&&Sentence two'".
- --voice: Use a voice recognition model to perform inference over an audio transcription. 
  If the 'record' string is given as an argument, it will prompt for a recording, otherwise 
  provide the path to the audio file as an argument.
- --twitter: Perform inference over tweets from the user specified as an argument.
- --reddit: Perform inference over comments from the user specified as an argument.
- --data\_path: Base path to the directory where all the pretrained models are stored, default=data/.

This information can be seen at any point by using the `--help` flag.

As mentioned in the flags, a 'data' directory is needed to run the ensemble. The directory
structure must be the same as the one provided in the repository's data directory.

## APIs

There are two APIs available: Twitter and Reddit.
The "--twitter" and "--reddit" flags require API keys loaded as environment variables,
and they take as an argument the name of a user whose tweets/comments will be analyzed.

To facilitate the management of credentials, we use the dotenv library, which allows the
loading of environment variables from a .env file located in the root of the project.

### Reddit API
Follow the instructions at the [reddit-archive][2] to get the credentials needed for the API use.
Once you have the information, you can then set the following environment variables in the .env file.
```bash
REDDIT_EMAIL=your@mail.com
REDDIT_USER=your_user
REDDIT_PASSWORD=your_password
REDDIT_CLIENT_ID=the_app_client_id
REDDIT_CLIENT_SECRET=the_app_client_secret
```

### Twitter API

To get this data you'll need a twitter developer account, after acquiring one you'll go to the
Developer Portal to the section Projects & Apps. You can create a Standalone App and select
"Keys and tokens" here you'll find all of the following keys and tokens.  

```bash
TWITTER_CONSUMER_KEY="consumer_key"
TWITTER_CONSUMER_SECRET="consumer_secret"
TWITTER_ACCESS_TOKEN="access_token"
TWITTER_ACCESS_SECRET="access_secret"
```


[1]: https://arxiv.org/abs/2005.12872
[2]: https://github.com/reddit-archive/reddit/wiki/OAuth2-Quick-Start-Example#first-steps

