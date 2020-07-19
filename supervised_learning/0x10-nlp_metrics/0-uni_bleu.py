#!/usr/bin/env python3
"""Evaluate a NLP traductor model"""

import numpy as np


def uni_bleu(references, sentence):
    """ calculates the unigram BLEU score for a sentence:

    Arg:
        - references: list of reference translations
            each reference translation list of the words
        - sentence: list containing the model proposed sentence

    Returns: the unigram BLEU score
    """
    output_len = len(sentence)
    count_clip = 0
    references_len = []
    counts_clip = {}
    for reference in references:
        references_len.append(len(reference))
        for word in reference:
            if word in sentence:
                if not counts_clip.keys() == word:
                    counts_clip[word] = 1

    count_clip = sum(counts_clip.values())

    reference_len = min(references_len, key=lambda x: abs(x-output_len))

    if output_len > reference_len:
        bp = 1
    else:
        bp = np.exp(1 - (reference_len / output_len))

    bleu_score = bp * np.exp(np.log(count_clip/output_len))

    return bleu_score
