import numpy as np
from similarity import fill_the_vector, build_text_matrix, find_most_similar


def test_fill_the_vector_counts_words():
    words = {"cat": 0, "dog": 1, "big": 2}
    sentence = "Cat dog cat big big big"
    vector = fill_the_vector(sentence, words)
    assert np.array_equal(vector, np.array([2, 1, 3]))


def test_distance_not_nan():
    sentences = ["Cats and dogs", "Dogs and cats"]
    matrix, _ = build_text_matrix(sentences)
    dist = np.linalg.norm(matrix[0] - matrix[1])  # alternative to avoid SciPy
    assert not np.isnan(dist)


def test_find_most_similar_sentences():
    with open('sentences.txt') as f:
        sentences = f.readlines()
    result = find_most_similar(sentences)
    assert result[0].startswith('Domestic cats are similar')
    assert result[1].startswith('In one, people deliberately tamed cats')
