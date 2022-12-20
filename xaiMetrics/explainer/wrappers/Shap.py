import shap
import pandas as pd

from xaiMetrics.constants import REFERENCE_FREE
from xaiMetrics.explainer.wrappers.AgnosticExplainer import AgnosticExplainer
from xaiMetrics.explanations.FeatureImportanceExplanation import FeatureImportanceExplanation

import numpy as np


class ShapExplainer(AgnosticExplainer):
    '''
    SHAP explainer for metrics, using the shap library https://github.com/slundberg/shap by:
    Scott M Lundberg and Su-In Lee. “A Unified Approach to Interpreting Model Predictions”. In: Advances
    in Neural Information Processing Systems 30. Ed. by I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach,
    R. Fergus, S. Vishwanathan, and R. Garnett. Curran Associates, Inc., 2017, pp. 4765–4774. url:
    http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions.pdf.

    The functions build_feature and masker are based on the implementation in:
    https://github.com/yg211/explainable-metrics
    '''

    def __init__(self, mask_string='UNKWORDZ'):
        '''
        :param mask_string: replacement string
        '''
        self.mask_string = mask_string

    def build_feature(self, sent1, sent2=None):
        '''
        Creates a pandas dataframe where each token of sent1 and optionally sent2 is written to a different column
        :param sent1: The tokens of the first sentence are prepended by s1_
        :param sent2: The tokens of the second sentence are prepended by s2_ . The second sentence is not used in the
                      final implementation
        :return:
        '''
        tdict = {}

        sent1_tokens = sent1.split(' ')
        self.l1len = len(sent1_tokens)

        for i in range(len(sent1_tokens)):
            tdict['s1_{}'.format(i)] = sent1_tokens[i]

        if sent2:
            sent2_tokens = sent2.split(' ')
            for i in range(len(sent2_tokens)):
                tdict['s2_{}'.format(i)] = sent2_tokens[i]
        return pd.DataFrame(tdict, index=[0])

    def masker(self, mask, x, sent2=False):
        '''
        replaces the tokens in x with a mask string, where indicated by a mask
        :param mask: A mask that indicates replacement positions
        :param x: Tokens of sentence 1
        :param sent2: If specified, it is assumed that x contains the tokens of sentence 2 as well. And they are processed
                      by putting [SEP] inbetween. This is not used in the final implementation
        :return:
        '''
        tokens = []
        for mm, tt in zip(mask, x):
            if mm:
                tokens.append(tt)
            else:
                tokens.append(self.mask_string)
        if sent2 == False:
            sentence_pair = ' '.join(tokens)
        else:
            s1 = ' '.join(tokens[self.l1len:])
            s2 = ' '.join(tokens[:self.l1len])
            sentence_pair = s1 + '[SEP]' + s2
        return pd.DataFrame([sentence_pair])

    def determine_method_and_features(self, sent, max_exact=7):
        '''
        Precomputes sentences and methods in input format
        :param sent: A hypothesis
        :param max_exact: The number of maximum features for which exact shap should be used
        :return: Dictionary with the method and pandas dataframes for each input sentence
        '''
        pre_dict = {'method': 'auto', 'hyp_features': self.build_feature(sent)}
        if len(sent.split()) <= max_exact:
            pre_dict['method'] = 'exact'
        return pre_dict

    def explain_sentence(self, gt, hyp, metric):
        '''
        Need to unwrap np array during metric calculation --> to list and afterwards wrap it back
        as features are precomputed, the actual hyp is gotten from pre_dict
        :param gt: dummy value for hypothesis (real hypothesis comes from pre_dict)
        :param metric: The metric to explain
        :return: explanation for the current sample
        '''
        gt_pre_dict = self.determine_method_and_features(gt)
        hyp_pre_dict = self.determine_method_and_features(hyp)
        explainer_gt = shap.Explainer(lambda x: np.array(metric([a[0] for a in x.tolist()], [hyp] * len(x))), self.masker,
                                      algorithm=gt_pre_dict['method'])
        explainer_hyp = shap.Explainer(lambda x: np.array(metric([gt] * len(x), [a[0] for a in x.tolist()])), self.masker,
                                      algorithm=hyp_pre_dict['method'])

        gt_attributions = explainer_gt(gt_pre_dict['hyp_features'])
        hyp_attributions = explainer_hyp(hyp_pre_dict['hyp_features'])

        return FeatureImportanceExplanation(score=metric([gt], [hyp]), gt=gt, hyp=hyp,
                                            gt_attributions=list(zip(gt_attributions[0].values, gt_attributions[0].data)),
                                            hyp_attributions=list(zip(hyp_attributions[0].values, hyp_attributions[0].data)))


if __name__ == '__main__':
    from xaiMetrics.metrics.wrappers.BertScore import BertScore

    shapex = ShapExplainer()
    print(
        shapex.explain_sentence("Ich habe einen kleinen Hund .", "I have a small book .",
                              BertScore(mode=REFERENCE_FREE)))
