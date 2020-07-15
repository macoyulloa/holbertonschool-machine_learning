#!/usr/bin/env python3
"""Word embedding models"""

import gensim


def fasttext_model(sentences, size=100, min_count=5, negative=5,
             window=5, cbow=True, iterations=5, seed=0):
    """ that creates and trains a genism fastText model:

    Arg:
        - sentences is a list of sentences to be trained on
        - size: dimensionality of the embedding layer
        - min_count: min num of occurrences of word for use in training
        - window: max distance between the current and predicted
            word within a sentence
        - negative: size of negative sampling
        - cbow: boolean to determine the training type; True is for CBOW;
            False is for Skip-gram
        - iterations: num of iterations to train over
        - seed: is the seed for the random number generator

    Returns: the trained model
    """
    model = gensim.models.FastText(sentences, min_count=min_count, iter=iterations,
                                   size=size, window=window,
                                   sg=cbow, seed=seed, negative=negative)

    model.train(sentences, total_examples=model.corpus_count,
                epochs=model.iter)

    return model
