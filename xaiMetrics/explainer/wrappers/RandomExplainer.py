import random

import tqdm

from xaiMetrics.constants import REFERENCE_FREE
from xaiMetrics.explainer.wrappers.AgnosticExplainer import AgnosticExplainer
from xaiMetrics.explanations.FeatureImportanceExplanation import FeatureImportanceExplanation


class RandomExplainer(AgnosticExplainer):
    def __init__(self):
        pass

    def explain_sentence(self, gt, hyp, metric):
        '''
        :param gt: ground truth sentence
        :param hyp: hypothesis sentence
        :param metric: metric to explain
        :return: list of tuples with feature importance and tokens, real score
        '''
        # compute the metric
        score = metric([gt], [hyp])

        # Determine the attribution of each token
        gt_attributions = [(random.uniform(0,1), g) for g in gt.split(" ")]
        hyp_attributions = [(random.uniform(0,1), h) for h in hyp.split(" ")]

        return FeatureImportanceExplanation(score, gt, hyp, gt_attributions, hyp_attributions, mode=metric.mode)


if __name__ == '__main__':
    from xaiMetrics.metrics.wrappers.BertScore import BertScore

    randomExp = RandomExplainer()
    print(
        randomExp.explain_sentence("Ich habe einen kleinen Hund .", "I have a small book .", BertScore(mode=REFERENCE_FREE)))