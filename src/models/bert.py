from pathlib import Path
import tensorflow as tf
import tensorflow_text as text
import zipfile
tf.get_logger().setLevel('ERROR')


def prepare_model(zipfile_path='data/bert/Equipo1BERT_bert-20210423T030536Z-001.zip'):
    zipfile_path = Path(zipfile_path)

    if not zipfile_path.exists():
        with zipfile.ZipFile(str(zipfile_path), 'r') as zip_ref:
            zip_ref.extractall('data/bert/')


def inference(inputs, path='data/bert/Equipo1BERT_bert'):
    classification=[]
    reloaded_model = tf.saved_model.load(path)
    results=tf.sigmoid(reloaded_model(tf.constant(inputs)))
    for i in range(len(inputs)):
    if float(results[i][0])<.50:
      classification.append(0)
    else:
      classification.append(1)
    return classification

