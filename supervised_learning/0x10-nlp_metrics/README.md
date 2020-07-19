0x10. Natural Language Processing - Evaluation Metrics
======================================================

Tasks
-----

#### 0\. Unigram BLEU score

Function `def uni_bleu(references, sentence):` that calculates the unigram BLEU score for a sentence:

-   `references` is a list of reference translations
    -   each reference translation is a list of the words in the translation
-   `sentence` is a list containing the model proposed sentence
-   Returns: the unigram BLEU score


#### 1\. N-gram BLEU score

Function `def ngram_bleu(references, sentence, n):` that calculates the n-gram BLEU score for a sentence:

-   `references` is a list of reference translations
    -   each reference translation is a list of the words in the translation
-   `sentence` is a list containing the model proposed sentence
-   `n` is the size of the n-gram to use for evaluation
-   Returns: the n-gram BLEU score


#### 2\. Cumulative N-gram BLEU score

Function `def cumulative_bleu(references, sentence, n):` that calculates the cumulative n-gram BLEU score for a sentence:

-   `references` is a list of reference translations
    -   each reference translation is a list of the words in the translation
-   `sentence` is a list containing the model proposed sentence
-   `n` is the size of the largest n-gram to use for evaluation
-   All n-gram scores should be weighted evenly
-   Returns: the cumulative n-gram BLEU score
