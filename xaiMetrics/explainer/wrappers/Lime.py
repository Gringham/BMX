import tqdm

from xaiMetrics.constants import REFERENCE_FREE, REFERENCE_BASED
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

    def __init__(self, split_expression=' ', class_names="score", num_samples=100):
        '''
        :param mask_string: replacement string
        '''
        self.explainer = LimeTextExplainer(class_names=[class_names, class_names], bow=False, split_expression=split_expression)
        self.num_samples = num_samples

    def explain_sentence(self, gt, hyp, metric):
        '''
        :param gt: ground truth sentence
        :param hyp: hypothesis sentence
        :param metric: metric to explain
        :return: list of tuples with feature importance and tokens, real score
        '''

        old_gt = gt
        if not type(gt) == list:
            gt = [gt]

        gt_attributions = []
        original = []
        for i, g in enumerate(gt):
            gt_split = g.split(' ')

            def gt_function(texts):
                texts_padded = [gt[:i] + [t] + gt[i+1:] if type(gt) == list and len(gt) > 1 else t for t in texts]
                scores = np.array(metric(texts_padded, [hyp] * len(texts_padded)))
                return np.vstack((scores, scores)).T

            gt_scores = self.explainer.explain_instance(g, gt_function, num_features=len(gt_split), labels=(1,),
                                                       num_samples=self.num_samples)
            gt_attributions += [(score, gt_split[i]) for i, score in sorted(gt_scores.as_map()[1], key=lambda x:x[0])]
            original += [gt_scores.predict_proba[0]]

        def hyp_function(hyp_i):
            if type(gt) == list and len(gt) > 1:
                scores = np.array(metric([gt] * len(hyp_i), hyp_i))
            else:
                scores = np.array(metric([old_gt] * len(hyp_i), hyp_i))
            return np.vstack((scores, scores)).T

        hyp_split = hyp.split(' ')
        hyp_scores = self.explainer.explain_instance(hyp, hyp_function, num_features=len(hyp_split), labels=(1,),
                                                   num_samples=self.num_samples)

        hyp_attributions = [(score, hyp_split[i]) for i, score in sorted(hyp_scores.as_map()[1], key=lambda x:x[0])]

        if not abs(original[0] - hyp_scores.predict_proba[0]) < 0.01:
            print("WARNING: unstable metric")

        gt = " ".join(gt)
        return FeatureImportanceExplanation(original[0], gt, hyp, gt_attributions, hyp_attributions, mode=metric.mode)


if __name__ == '__main__':
    from xaiMetrics.metrics.wrappers.BertScore import BertScore

    lime = LimeExplainer()
    print(
        lime.explain_sentence("Ich habe einen kleinen Hund .", "I have a small book .",
                                 BertScore(mode=REFERENCE_FREE)))

    print(
        lime.explain_sentence(["Ich habe einen kleinen Hund .","Mein Hund ist sehr klein", "Wieso ist mein Hund so klein?"], "Ich habe keinen kleinen Hund .",
                              BertScore(mode=REFERENCE_BASED)))
    print(
        lime.explain_sentence(
            ["Mein Hund heißt Muffin.", "Muffin heißt mein Hund", "Der Name meines Hundes ist Muffin"],
            "Ich habe einen Hund mit Namen Muffin",
            BertScore(mode=REFERENCE_BASED)))

    b = BertScore(mode=REFERENCE_BASED)
    print(b([["Ich habe einen kleinen Hund .","Mein Hund ist sehr klein", "Wieso ist mein Hund so klein?"],
             ["Mein Hund heißt Muffin.", "Muffin heißt mein Hund", "Der Name meines Hundes ist Muffin"]],
            ["Ich habe keinen kleinen Hund .", "Ich habe einen Hund mit Namen Muffin"]))
