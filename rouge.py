"""Computes rouge scores between two text or two list of text.
Implemented based on https://github.com/google-research/google-research/tree/master/rouge
"""
import re
import collections
import numpy as np

import six
from six.moves import map
from six.moves import range


class Score(collections.namedtuple("Score", ["precision", "recall", "fmeasure"])):
    """Tuple containing precision, recall, and f-measure values."""


class AggregateScore(collections.namedtuple("AggregateScore", ["low", "mid", "high"])):
    """Tuple containing confidence intervals for scores."""


class BootstrapAggregator(object):
    """Aggregates scores to provide confidence intervals.

    Sample usage:
      scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'])
      aggregator = Aggregator()
      aggregator.add_scores(scorer.score("one two three", "one two"))
      aggregator.add_scores(scorer.score("one two five six", "seven eight"))
      result = aggregator.aggregate()
      print result
      {'rougeL': AggregateScore(
           low=Score(precision=0.0, recall=0.0, fmeasure=0.0),
           mid=Score(precision=0.5, recall=0.33, fmeasure=0.40),
           high=Score(precision=1.0, recall=0.66, fmeasure=0.80)),
       'rouge1': AggregateScore(
           low=Score(precision=0.0, recall=0.0, fmeasure=0.0),
           mid=Score(precision=0.5, recall=0.33, fmeasure=0.40),
           high=Score(precision=1.0, recall=0.66, fmeasure=0.80))}
    """

    def __init__(self, confidence_interval=0.95, n_samples=1000):
        """Initializes a BootstrapAggregator object.

        Args:
          confidence_interval: Confidence interval to compute on the mean as a
            decimal.
          n_samples: Number of samples to use for bootstrap resampling.

        Raises:
          ValueError: If invalid argument is given.
        """

        if confidence_interval < 0 or confidence_interval > 1:
            raise ValueError("confidence_interval must be in range [0, 1]")
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")

        self._n_samples = n_samples
        self._confidence_interval = confidence_interval
        self._scores = collections.defaultdict(list)

    def add_scores(self, scores):
        """Adds a sample for future aggregation.

        Args:
          scores: Dict mapping score_type strings to a namedtuple object/class
            representing a score.
        """

        for score_type, score in six.iteritems(scores):
            self._scores[score_type].append(score)

    def aggregate(self):
        """Aggregates scores previously added using add_scores.

        Returns:
          A dict mapping score_type to AggregateScore objects.
        """

        result = {}
        for score_type, scores in six.iteritems(self._scores):
            # Stack scores into a 2-d matrix of (sample, measure).
            score_matrix = np.vstack(tuple(scores))
            # Percentiles are returned as (interval, measure).
            percentiles = self._bootstrap_resample(score_matrix)
            # Extract the three intervals (low, mid, high).
            intervals = tuple(
                (scores[0].__class__(*percentiles[j, :]) for j in range(3)))
            result[score_type] = AggregateScore(
                low=intervals[0], mid=intervals[1], high=intervals[2])
        return result

    def _bootstrap_resample(self, matrix):
        """Performs bootstrap resampling on a matrix of scores.

        Args:
          matrix: A 2-d matrix of (sample, measure).

        Returns:
          A 2-d matrix of (bounds, measure). There are three bounds: low (row 0),
          mid (row 1) and high (row 2). Mid is always the mean, while low and high
          bounds are specified by self._confidence_interval (which defaults to 0.95
          meaning it will return the 2.5th and 97.5th percentiles for a 95%
          confidence interval on the mean).
        """

        # Matrix of (bootstrap sample, measure).
        sample_mean = np.zeros((self._n_samples, matrix.shape[1]))
        for i in range(self._n_samples):
            sample_idx = np.random.choice(
                np.arange(matrix.shape[0]), size=matrix.shape[0])
            sample = matrix[sample_idx, :]
            sample_mean[i, :] = np.mean(sample, axis=0)

        # Take percentiles on the estimate of the mean using bootstrap samples.
        # Final result is a (bounds, measure) matrix.
        percentile_delta = (1 - self._confidence_interval) / 2
        q = 100 * np.array([percentile_delta, 0.5, 1 - percentile_delta])
        return np.percentile(sample_mean, q, axis=0)


class RougeScorer:
    """Calculate rouges scores between two blobs of text.

    Sample usage:
      scorer = RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
      scores = scorer.score('The quick brown fox jumps over the lazy dog',
                            'The quick brown dog jumps on the log.')
    """

    def __init__(self, rouge_types):
        """Initializes a new RougeScorer.

        Valid rouge types that can be computed are:
          rougen (e.g. rouge1, rouge2): n-gram based scoring.
          rougeL: Longest common subsequence based scoring.

        Args:
          rouge_types: A list of rouge types to calculate.
        Returns:
          A dict mapping rouge types to Score tuples.
        """

        self.rouge_types = rouge_types

    @staticmethod
    def _create_ngrams(tokens, n):
        """Creates ngrams from the given list of tokens.

        Args:
          tokens: A list of tokens from which ngrams are created.
          n: Number of tokens to use, e.g. 2 for bigrams.
        Returns:
          A dictionary mapping each bigram to the number of occurrences.
        """

        ngrams = collections.Counter()
        for ngram in (tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)):
            ngrams[ngram] += 1
        return ngrams

    @staticmethod
    def _lcs_table(ref, can):
        """Create 2-d LCS score table."""
        rows = len(ref)
        cols = len(can)
        lcs_table = [[0] * (cols + 1) for _ in range(rows + 1)]
        for i in range(1, rows + 1):
            for j in range(1, cols + 1):
                if ref[i - 1] == can[j - 1]:
                    lcs_table[i][j] = lcs_table[i - 1][j - 1] + 1
                else:
                    lcs_table[i][j] = max(lcs_table[i - 1][j], lcs_table[i][j - 1])
        return lcs_table

    @staticmethod
    def _backtrack_norec(t, ref, can):
        """Read out LCS."""
        i = len(ref)
        j = len(can)
        lcs = []
        while i > 0 and j > 0:
            if ref[i - 1] == can[j - 1]:
                lcs.insert(0, i - 1)
                i -= 1
                j -= 1
            elif t[i][j - 1] > t[i - 1][j]:
                j -= 1
            else:
                i -= 1
        return lcs

    def lcs_ind(self, ref, can):
        """Returns one of the longest lcs."""
        t = self._lcs_table(ref, can)
        return self._backtrack_norec(t, ref, can)

    @staticmethod
    def _find_union(lcs_list):
        """Finds union LCS given a list of LCS."""
        return sorted(list(set().union(*lcs_list)))

    def _union_lcs(self, ref, c_list):
        """Find union LCS between a ref sentence and list of candidate sentences.

        Args:
          ref: list of tokens
          c_list: list of list of indices for LCS into reference summary

        Returns:
          List of tokens in ref representing union LCS.
        """
        lcs_list = [self.lcs_ind(ref, c) for c in c_list]
        return [ref[i] for i in self._find_union(lcs_list)]

    def _summary_level_lcs(self, ref_sent, can_sent):
        """ROUGE: Summary-level LCS, section 3.2 in ROUGE paper.

        Args:
          ref_sent: list of tokenized reference sentences
          can_sent: list of tokenized candidate sentences

        Returns:
          summary level ROUGE score
        """
        if not ref_sent or not can_sent:
            return Score(precision=0, recall=0, fmeasure=0)

        m = sum(map(len, ref_sent))
        n = sum(map(len, can_sent))
        if not n or not m:
            return Score(precision=0, recall=0, fmeasure=0)

        # get token counts to prevent double counting
        token_cnts_r = collections.Counter()
        token_cnts_c = collections.Counter()
        for s in ref_sent:
            # s is a list of tokens
            token_cnts_r.update(s)
        for s in can_sent:
            token_cnts_c.update(s)

        hits = 0
        for r in ref_sent:
            lcs = self._union_lcs(r, can_sent)
            # Prevent double-counting:
            # The paper describes just computing hits += len(_union_lcs()),
            # but the implementation prevents double counting. We also
            # implement this as in version 1.5.5.
            for t in lcs:
                if token_cnts_c[t] > 0 and token_cnts_r[t] > 0:
                    hits += 1
                    token_cnts_c[t] -= 1
                    token_cnts_r[t] -= 1

        recall = hits / m
        precision = hits / n
        fmeasure = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        return Score(precision=precision, recall=recall, fmeasure=fmeasure)

    def _score_lcs(self, target_tokens, prediction_tokens):
        """Computes LCS (Longest Common Subsequence) rouge scores.

        Args:
          target_tokens: Tokens from the target text.
          prediction_tokens: Tokens from the predicted text.
        Returns:
          A Score object containing computed scores.
        """

        if not target_tokens or not prediction_tokens:
            return Score(precision=0, recall=0, fmeasure=0)

        # Compute length of LCS from the bottom up in a table (DP appproach).
        lcs_table = self._lcs_table(target_tokens, prediction_tokens)
        lcs_length = lcs_table[-1][-1]

        precision = lcs_length / len(prediction_tokens)
        recall = lcs_length / len(target_tokens)
        fmeasure = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

        return Score(precision=precision, recall=recall, fmeasure=fmeasure)

    @staticmethod
    def _score_ngrams(target_ngrams, prediction_ngrams):
        """Compute n-gram based rouge scores.

        Args:
          target_ngrams: A Counter object mapping each ngram to number of
            occurrences for the target text.
          prediction_ngrams: A Counter object mapping each ngram to number of
            occurrences for the prediction text.
        Returns:
          A Score object containing computed scores.
        """

        intersection_ngrams_count = 0
        for ngram in six.iterkeys(target_ngrams):
            intersection_ngrams_count += min(target_ngrams[ngram],
                                             prediction_ngrams[ngram])
        target_ngrams_count = sum(target_ngrams.values())
        prediction_ngrams_count = sum(prediction_ngrams.values())

        precision = intersection_ngrams_count / max(prediction_ngrams_count, 1)
        recall = intersection_ngrams_count / max(target_ngrams_count, 1)
        fmeasure = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

        return Score(precision=precision, recall=recall, fmeasure=fmeasure)

    def score(self, target, prediction):
        """Calculates rouge scores between the target and prediction.

        Args:
          target: Text containing the target (ground truth) text.
          prediction: Text containing the predicted text.
        Returns:
          A dict mapping each rouge type to a Score object.
        Raises:
          ValueError: If an invalid rouge type is encountered.
        """

        target_tokens = re.split(r"\s+", target)
        prediction_tokens = re.split(r"\s+", prediction)
        result = {}

        for rouge_type in self.rouge_types:
            if rouge_type == "rougeL":
                # Rouge from longest common subsequences.
                result[rouge_type] = self._score_lcs(target_tokens, prediction_tokens)
            elif rouge_type == "rougeLsum":
                # Note: Does not support multi-line text.
                def get_sents(text):
                    # Assume sentences are separated by newline.
                    sentences = six.ensure_str(text).split("\n")
                    sentences = [x for x in sentences if len(x)]
                    return sentences

                target_tokens_list = [re.split(r"\s+", s) for s in get_sents(target)]
                prediction_tokens_list = [re.split(r"\s+", s) for s in get_sents(prediction)]
                result[rouge_type] = self._summary_level_lcs(target_tokens_list, prediction_tokens_list)
            elif re.match(r"rouge[0-9]$", six.ensure_str(rouge_type)):
                # Rouge from n-grams.
                n = int(rouge_type[5:])
                if n <= 0:
                    raise ValueError("rougen requires positive n: %s" % rouge_type)
                target_ngrams = self._create_ngrams(target_tokens, n)
                prediction_ngrams = self._create_ngrams(prediction_tokens, n)
                result[rouge_type] = self._score_ngrams(target_ngrams, prediction_ngrams)
            else:
                raise ValueError("Invalid rouge type: %s" % rouge_type)

        return result

    def compute(self, predictions, references):
        """Calculates average rouge scores for a list of hypotheses and references

        Args:
          predictions: List of predictions to score. Each predictions should be a string with tokens separated by
          spaces.
          references: List of reference for each prediction. Each reference should be a string with tokens
          separated by spaces.
        Returns:
          Aggregated scores
        """

        assert len(references) == len(predictions), "Length of references and predictions must be equal!"

        aggregator = BootstrapAggregator()
        for ref, pred in zip(references, predictions):
            score = self.score(ref, pred)
            aggregator.add_scores(score)
        result = aggregator.aggregate()

        return result
