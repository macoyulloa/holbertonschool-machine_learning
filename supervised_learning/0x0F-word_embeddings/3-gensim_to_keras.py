#!/usr/bin/env python3
"""Word embedding models"""


def gensim_to_keras(model):
    """converts the gensim word2vec model to a keras layer:

    Requirements:
        - keras version 2.2.2 needs to be install to succes,
          with tesorflow as backend.
              pip3 install heras=2.2.2

    Arg:
        - model: is a trained gensim word2vec models

    Returns: the trainable keras Embedding
    """
    return model.wv.get_keras_embedding(train_embeddings=False)
