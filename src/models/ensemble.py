from src import models
from src.models.text_preprocess import preprocess


# TODO: deal with sklearn's np.nan as a value
def inference(sentences, data_path='data/', return_inferences=True):
    print('Processing text...')
    processed_sentences = [preprocess(s) for s in sentences]

    predictions = {}

    print('Performing inference with rnns...')
    rnn_predictions = models.rnn.inference(processed_sentences, data_path)
    predictions.update(rnn_predictions)

    print('Performing inference with sklearn models...')
    sklearn_predictions = models.sklearn.inference(processed_sentences, data_path)
    predictions.update(sklearn_predictions)

    print('Performing inference with BERT...')
    bert_predictions = models.bert.inference(processed_sentences, data_path)
    predictions.update(bert_predictions)

    result = combine_predictions(predictions)

    if return_inferences:
        result = (result, predictions)

    return result


def combine_predictions(predictions, positive_threshold=0.5):
    _check_same_length(predictions)

    final_predictions = []
    for models_prediction in zip(*predictions.values()):
        avg = sum(models_prediction) / len(models_prediction)
        if avg >= positive_threshold:
            final_predictions.append(1)
        else:
            final_predictions.append(0)

    return final_predictions


def _check_same_length(predictions):
    n = len(next(predictions.values().__iter__()))

    same_length = True
    for value in predictions.values():
        if len(value) != n:
            same_length = False

    if not same_length:
        raise ValueError("Not all predictions are the same length.")
