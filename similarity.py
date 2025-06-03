import re
import numpy as np
from scipy.spatial.distance import cosine


def fill_the_vector(line: str, words_dict: dict) -> np.ndarray:
    """Return a vector of word counts for `line` based on `words_dict`."""
    vector = np.zeros(len(words_dict), dtype=int)
    words = re.split('[^a-z]', line.lower())
    words = list(filter(None, words))
    for word in words:
        if word in words_dict:
            vector[words_dict[word]] += 1
    return vector


def build_text_matrix(sentences):
    """Build matrix of word counts for each sentence and return (matrix, vocab)."""
    all_words = []
    for sent in sentences:
        all_words += re.split('[^a-z]', sent.lower())
    all_words = list(filter(None, all_words))
    words_dict = {}
    for word in all_words:
        if word not in words_dict:
            words_dict[word] = len(words_dict)
    matrix = np.zeros((len(sentences), len(words_dict)), dtype=int)
    for i, sent in enumerate(sentences):
        matrix[i] = fill_the_vector(sent, words_dict)
    return matrix, words_dict


def find_most_similar(sentences, index=0, top_n=2):
    """Return the `top_n` sentences most similar to sentence at `index`."""
    matrix, _ = build_text_matrix(sentences)
    base_vector = matrix[index]
    distances = [cosine(base_vector, vec) for vec in matrix]
    ranked = np.argsort(distances)
    result = []
    for idx in ranked:
        if idx == index:
            continue
        result.append(sentences[idx].strip())
        if len(result) == top_n:
            break
    return result
