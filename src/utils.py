import pickle
from pathlib import Path


def load_from_pickle(path):
    path = Path(path)
    if not path.exists():
        raise AttributeError(f"The file {str(path)} doesn't exist.")

    with open(path, 'rb') as f:
        result = pickle.load(f)

    return result


def interpret_results(results, threshold=0.5):
    return [0 if r < threshold else 1 for r in results]

