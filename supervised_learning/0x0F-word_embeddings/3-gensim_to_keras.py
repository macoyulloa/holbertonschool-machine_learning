#!/usr/bin/env python3
"""Word embedding models"""

import gensim
import tensorflow as tf


def gensim_to_keras(model):
    """converts the gensim word2vec model to a keras layer:

    Arg:
        - model: is a trained gensim word2vec models

    Returns: the trainable keras Embedding
    """
    print(model)
    return tf.keras.layers.Embedding(input_dim=model.corpus_count,
                                     output_dim=model.size)
