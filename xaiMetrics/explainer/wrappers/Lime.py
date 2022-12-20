import tqdm

from xaiMetrics.constants import REFERENCE_FREE
from xaiMetrics.explainer.wrappers.AgnosticExplainer import AgnosticExplainer
from xaiMetrics.explanations.FeatureImportanceExplanation import FeatureImportanceExplanation

from lime.lime_text import LimeTextExplainer
import numpy as np


class LimeExplainer(AgnosticExplainer):
    '''
    Using LIME to generate feature importance explanations for metrics. LIME was proposed by
    Marco Tulio Ribeiro, Sameer Singh, and Carlos Guestrin. “”Why Should I Trust You?”: Explaining the
    Predictions of Any Classifier”. In: Kdd ’16. San Francisco, California, USA: Association for Computing
    Machinery, 2016, pp. 1135–1144. isbn: 9781450342322. doi: 10.1145/2939672.2939778. url:
    https://doi.org/10.1145/2939672.2939778.
    '''

    def __init__(self, mask_string=' ', class_names="score", num_samples=100):
        '''
        :param mask_string: replacement string
        '''
        self.explainer = LimeTextExplainer(class_names=[class_names, class_names], bow=False, split_expression=mask_string)
        self.num_samples = num_samples

    def explain_sentence(self, gt, hyp, metric):
        '''
        :param gt: ground truth sentence
        :param hyp: hypothesis sentence
        :param metric: metric to explain
        :return: list of tuples with feature importance and tokens, real score
        '''

        def gt_function(texts):
            scores = np.array(metric(texts, [hyp]*len(texts)))
            return np.vstack((scores,scores)).T

        gt_split = gt.split()
        gt_scores = self.explainer.explain_instance(gt, gt_function, num_features=len(gt_split), labels=(1,),
                                                   num_samples=self.num_samples)
        gt_attributions = [(score, gt_split[i]) for i, score in sorted(gt_scores.as_map()[1], key=lambda x:x[0])]
        original = gt_scores.predict_proba[0]

        def hyp_function(hyp_i):
            scores = np.array(metric([gt] * len(hyp_i), hyp_i))
            return np.vstack((scores, scores)).T

        hyp_split = hyp.split()
        hyp_scores = self.explainer.explain_instance(hyp, hyp_function, num_features=len(hyp_split), labels=(1,),
                                                   num_samples=self.num_samples)
        hyp_attributions = [(score, hyp_split[i]) for i, score in sorted(hyp_scores.as_map()[1], key=lambda x:x[0])]

        if not abs(original - hyp_scores.predict_proba[0]) < 0.01:
            print("WARNING: unstable metric")
        return FeatureImportanceExplanation(original, gt, hyp, gt_attributions, hyp_attributions, mode=metric.mode)


if __name__ == '__main__':
    from xaiMetrics.metrics.wrappers.BertScore import BertScore

    lime = LimeExplainer()
    print(
        lime.explain_sentence("Ich habe einen kleinen Hund .", "I have a small book .",
                                 BertScore(mode=REFERENCE_FREE)))
