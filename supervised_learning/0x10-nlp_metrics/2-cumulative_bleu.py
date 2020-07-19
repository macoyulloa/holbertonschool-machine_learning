#!/usr/bin/env python3
"""Evaluate a NLP traductor model"""

import numpy as np


def ngram_bleu(references, sentence, n, n_precissions=[], reference_len=0):
    """ calculates the unigram BLEU score for a sentence:

    Arg:
        - references: list of reference translations
            each reference translation list of the words
        - sentence: list containing the model proposed sentence
        - n is the size of the n-gram to use for evaluation
        - n_precissions: accumulate of the n-grams until n
        - reference_len: the len of the refenrece sentence
                        closest to sentence len

    Returns: the unigram BLEU score
    """
    output_len = len(sentence)
    count_clip = 0
    references_len = []
    counts_clip = {}

    if n != 0:
        # n-sentence pass to the grams that we need
        n_sentence = [' '.join([str(j) for j in sentence[i:i+n]])
                      for i in range(len(sentence)-(n-1))]
        # n_sentence = [(str(sentence[i]) + ' ' + str(sentence[i+1]))
        #              for i in range(len(sentence)-(n-1))]
        n_output_len = len(n_sentence)

        for reference in references:
            n_reference = [' '.join([str(j) for j in reference[i:i+n]])
                           for i in range(len(sentence)-(n-1))]
            # n_reference = [(str(reference[i]) + ' ' + str(reference[i+1]))
            #               for i in range(len(reference)-(n-1))]

            references_len.append(len(reference))
            for word in n_reference:
                if word in n_sentence:
                    if not counts_clip.keys() == word:
                        counts_clip[word] = 1

        # gett the count clips that the sentences has in the references
        count_clip = sum(counts_clip.values())

        # get the less distante of the output len, of the sentence
        reference_len = min(references_len, key=lambda x: abs(x-output_len))

        precission = (np.log(count_clip/n_output_len))

        n_precissions.append(precission)
        return (ngram_bleu(references, sentence, n-1,
                           n_precissions, reference_len))

    else:
        return n_precissions, reference_len


def cumulative_bleu(references, sentence, n):
    """ calculates the unigram BLEU score for a sentence:

    Arg:
        - references: list of reference translations
            each reference translation list of the words
        - sentence: list containing the model proposed sentence
        - n is the size of the n-gram to use for evaluation

    Returns: the unigram BLEU score
    """
    output_len = len(sentence)

    # getting the array with the precission of each n-gram until n
    n_precissions, reference_len = ngram_bleu(references, sentence, n)
    n_precissions = np.asarray(n_precissions)
    n_precissions /= n
    precission = np.exp(np.sum(n_precissions))

    # find the bp of the model, breverty penalty of the model
    if output_len > reference_len:
        bp = 1
    else:
        bp = np.exp(1 - (reference_len / output_len))

    bleu_score = bp * precission

    return bleu_score
