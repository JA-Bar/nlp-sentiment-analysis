from pathlib import Path
import tensorflow as tf
import tensorflow_text as text
import zipfile
tf.get_logger().setLevel('ERROR')


def prepare_model(data_path='data/', zipfile_name='Equipo1BERT_bert-20210423T030536Z-001.zip'):
    zipfile_path = Path(data_path, 'bert', zipfile_name)

    if not zipfile_path.with_name('Equipo1BERT_bert').exists():
        with zipfile.ZipFile(str(zipfile_path), 'r') as zip_ref:
            zip_ref.extractall('data/bert/')


def inference(inputs, data_path='data', model_name='Equipo1BERT_bert'):
    reloaded_model = tf.saved_model.load(str(Path(data_path, 'bert', model_name)))
    results = tf.sigmoid(reloaded_model(tf.constant(inputs)))

    classification = []
    for i in range(len(inputs)):
        if float(results[i][0]) < 0.50:
            classification.append(0)
        else:
            classification.append(1)

    return classification

