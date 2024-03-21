from seaborn import FacetGrid

from xaiMetrics.evalTools.utils.power_mean import power_mean
from typing import Union
import numpy as np


class FeatureImportanceExplanation:
    def __init__(self, score, gt: str, hyp: str, gt_attributions: list[tuple[float, str]],
                 hyp_attributions: list[tuple[float, str]],
                 mode: str = "ref-free"):
        """
        :param gt_attributions:
        A list of Tuples with (AttributionScore, Token/Word). For each token/word, the attribution score should
        describe its importance. A higher AttributionScore describes that the word has a stronger influence on the
        current prediction result.
        :param hyp_attributions:
        A list of Tuples with (AttributionScore, Token/Word). For each token/word, the attribution score should
        describe its importance.
        :param mode: Whether the current explanation is "ref-free" or "ref-based"
        """
        self.score = score
        self.gt_attributionScores, self.gt_attributionTokens = map(list, zip(*gt_attributions))
        self.hyp_attributionScores, self.hyp_attributionTokens = map(list, zip(*hyp_attributions))
        self.gt_attributionScores = np.array(self.gt_attributionScores)
        self.hyp_attributionScores = np.array(self.hyp_attributionScores)
        self.concat = np.concatenate((self.gt_attributionScores, self.hyp_attributionScores))
        self.mode = mode
        self.gt = gt
        self.hyp = hyp

    def invert(self):
        """
        Inverts the current attribution scores.
        """
        self.gt_attributionScores = -self.gt_attributionScores
        self.hyp_attributionScores = -self.hyp_attributionScores

    def __str__(self) -> str:
        return """
        ------------------------------------------------
        Ground Truth: {GT}
        Hypothesis: {HYP}
        Metric Score: {SCORE}
        Mode: {MODE}
        Ground Truth Tokens:t {GTTOKEN}
        Ground Truth Attributions {GTSCORES}
        Hypothesis Tokens: {HYPTOKEN} 
        Hypothesis Attributions: {HYPSCORES}
        ------------------------------------------------
        """.format(GT=self.gt, HYP=self.hyp, SCORE=self.score, MODE=self.mode, GTTOKEN=self.gt_attributionTokens,
                   GTSCORES=self.gt_attributionScores, HYPTOKEN=self.hyp_attributionTokens,
                   HYPSCORES=self.hyp_attributionScores)


class FeatureImportanceExplanationCollection:
    """
    A class to hold multiple feature importance explanations and efficiently compute the powermean on all of them
    """

    def __init__(self, featureImportanceExplanations: Union[list[FeatureImportanceExplanation], None]):
        if not featureImportanceExplanations:
            self.featureImportanceExplanations = []
        else:
            self.featureImportanceExplanations = featureImportanceExplanations

        self.scores = np.array([f.score for f in self.featureImportanceExplanations])
        if self.scores.ndim >= 2:
            self.scores = self.scores[:, 0]

        self.refs = None

    def unique_mode(self) -> bool:
        return len(set([f.mode for f in self.featureImportanceExplanations])) == 1

    def apply_pmean(self, p, only_hyp, normalize=False, first_ref=False):
        if only_hyp:
            return np.array(
                list(map(lambda vals: power_mean(vals, p),
                         [f.hyp_attributionScores for f in self.featureImportanceExplanations])))
        if normalize:
            ret = np.array(
                list(map(lambda vals: power_mean(vals, p),
                         [f.concat for f in self.featureImportanceExplanations])))
            return ret.mean() / ret.std()

        if first_ref:
            if not self.refs:
                raise Exception("No references given to match in first ref mode")
            else:
                lens = [len(r[0].split(" ")) for r in self.refs]

            return np.array(
                list(map(lambda vals: power_mean(vals, p),
                         [np.concatenate((f.gt_attributionScores[:l], f.hyp_attributionScores)) for f,l in
                          zip(self.featureImportanceExplanations,lens)])))

        return np.array(
            list(map(lambda vals: power_mean(vals, p), [f.concat for f in self.featureImportanceExplanations])))

    def apply_pmean_multi(self, p_list, only_hyp, normalize=False, first_ref=False):
        return np.array([self.apply_pmean(p, only_hyp, normalize, first_ref=first_ref) for p in p_list])

    def combine_pmean_with_scores(self, w_list, p_list, only_hyp, normalize=False, first_ref=False):
        p_mean_matrix = self.apply_pmean_multi(p_list, only_hyp, normalize, first_ref=first_ref)
        w_list = np.array(w_list)
        weighted_orig = w_list * np.repeat(self.scores[:, np.newaxis], len(w_list), axis=1)
        weighted_expl = (1 - w_list) * np.repeat(p_mean_matrix[:, :, np.newaxis], len(w_list), axis=2)
        new_scores = np.repeat(weighted_orig[np.newaxis, :, :], len(p_list), axis=0) + weighted_expl
        w_indices = np.repeat(w_list[np.newaxis, np.newaxis, :], new_scores.shape[0], axis=0).ravel().tolist()
        p_indices = np.repeat(np.array(p_list)[np.newaxis, np.newaxis, :], new_scores.shape[2], axis=2).ravel().tolist()
        return new_scores, w_indices, p_indices

    def select_range(self, subset):
        if len(subset) > 2:
            return FeatureImportanceExplanationCollection(self.featureImportanceExplanations[
                                                          subset[0]:subset[1]] + self.featureImportanceExplanations[
                                                                                 subset[2]:subset[3]])
        else:
            return FeatureImportanceExplanationCollection(self.featureImportanceExplanations[subset[0]:subset[1]])

    def test_hyp(self, hyp: list[str]):
        for a, b in zip(hyp, [f.hyp for f in self.featureImportanceExplanations]):
            if a != b:
                raise Exception("Your dataframe was not synced in HYP order", a, b)

    def __str__(self):
        return "\n".join([str(f) for f in self.featureImportanceExplanations])


if __name__ == '__main__':
    f1 = FeatureImportanceExplanation(score=0,
                                      gt="dummy",
                                      hyp="dummy",
                                      gt_attributions=[(0.1, ""), (0.3, ""), (0.2, ""), (0.15, ""), (0.2, ""),
                                                       (0.1, "")],
                                      hyp_attributions=[(0.2, ""), (0.3, ""), (0.1, ""), (0.15, ""), (0.2, ""),
                                                        (0.1, "")])

    f2 = FeatureImportanceExplanation(score=0,
                                      gt="dummy",
                                      hyp="dummy",
                                      gt_attributions=[(0.1, ""), (0.3, ""), (0.2, ""), (0.15, ""), (0.2, ""),
                                                       (0.1, "")],
                                      hyp_attributions=[(0.2, ""), (0.3, ""), (0.1, ""), (0.15, ""), (0.2, ""),
                                                        (0.1, "")])

    fc = FeatureImportanceExplanationCollection([f1, f2])

    print(fc.combine_pmean_with_scores([0, 1], [-300, -1, 1, 2, 300], only_hyp=False))
