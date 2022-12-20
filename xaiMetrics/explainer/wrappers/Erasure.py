import tqdm

from xaiMetrics.constants import REFERENCE_FREE
from xaiMetrics.explainer.wrappers.AgnosticExplainer import AgnosticExplainer
from xaiMetrics.explanations.FeatureImportanceExplanation import FeatureImportanceExplanation, \
    FeatureImportanceExplanationCollection


class ErasureExplainer(AgnosticExplainer):
    '''
    Simple explainer that will erase tokens one by one and measure the effect on the metric.
    '''

    def __init__(self, mask_string=''):
        '''
        :param mask_string: replacement string
        '''
        self.mask_string = mask_string

    def permute_sentence(self, gt, hyp):
        # Precompute the sentences with all positions masked out one after another
        gt_splt = gt.split()
        hyp_splt = hyp.split()

        gt_permutations = [[gt_splt[y] if y != x else self.mask_string for y in range(len(gt_splt))] for x in
                           range(len(gt_splt))]
        hyp_permutations = [[hyp_splt[y] if y != x else self.mask_string for y in range(len(hyp_splt))] for x in
                            range(len(hyp_splt))]
        gt_permutations_2 = [' '.join(s) for s in gt_permutations]
        hyp_permutations_2 = [' '.join(s) for s in hyp_permutations]

        hyp_permutations = [hyp] + [hyp] * len(gt_permutations_2) + hyp_permutations_2
        gt_permutations = [gt] + gt_permutations_2 + [gt] * len(hyp_permutations_2)
        return (gt_permutations, len(gt_permutations_2), gt_splt), (hyp_permutations, len(hyp_permutations_2), hyp_splt)

    def explain_sentence(self, gt, hyp, metric):
        '''
        :param gt: ground truth sentence
        :param hyp: hypothesis sentence
        :param metric: metric to explain
        :return: list of tuples with feature importance and tokens, real score
        '''
        gt_permutations, hyp_permutations = self.permute_sentence(gt, hyp)

        # compute the metric
        scores = metric(gt_permutations[0], hyp_permutations[0])
        gt_scores = scores[1: 1 + gt_permutations[1]]
        hyp_scores = scores[1 + gt_permutations[1]:]

        # Determine the attribution of each token
        gt_attributions = [(scores[0] - gt_scores[x], gt_permutations[2][x]) for x in range(len(gt_scores))]
        hyp_attributions = [(scores[0] - hyp_scores[x], hyp_permutations[2][x]) for x in range(len(hyp_scores))]

        return FeatureImportanceExplanation(scores[0], gt, hyp, gt_attributions, hyp_attributions, mode=metric.mode)


if __name__ == '__main__':
    from xaiMetrics.metrics.wrappers.BertScore import BertScore

    erasure = ErasureExplainer()
    print(
        erasure.explain_sentence("Ich habe einen kleinen Hund .", "I have a small book .", BertScore(mode=REFERENCE_FREE)))