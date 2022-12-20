from sacremoses import MosesDetokenizer
import truecase

import numpy as np

from xaiMetrics.constants import REFERENCE_FREE
from xaiMetrics.metrics.utils.metricState import metricState
from xaiMetrics.metrics.wrappers.MetricClass import MetricClass
from xaiMetrics.metrics.wrappers.source.scorer import XMOVERScorer


class XMoverScore(MetricClass):
    '''A wrapper for XMoverScore (https://github.com/AIPHES/ACL20-Reference-Free-MT-Evaluation), by:
    Wei Zhao, Goran Glavaš, Maxime Peyrard, Yang Gao, Robert West, and Steffen Eger. “On the Lim-
    itations of Cross-lingual Encoders as Exposed by Reference-Free Machine Translation Evaluation”.
    In: Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics. Online:
    Association for Computational Linguistics, July 2020, pp. 1656–1671.
    url: https://www.aclweb.org/anthology/2020.acl-main.151 . This wrapper uses no mapping.'''
    ref_based = False
    name = 'XMOVERSCORE'

    def __init__(self, bs=16, layer=8, n_gram=1, model_name='bert-base-multilingual-cased', lp=None):
        '''
        :param bs: batch size
        :param layer: layer, only use this parameter with the xlm version
        :param model_name: The model name for the xlmr model, when xlm mode is used
        :param extension:
        '''
        self.bs = bs
        self.model_name = model_name
        self.scorer = XMOVERScorer(model_name, 'gpt2', False)
        self.n_gram = n_gram
        self.layer = layer
        self.lp = lp
        self.version = "XMOVER-V2-June-2022-No-Mapping"
        self.mode = REFERENCE_FREE

    def __call__(self, gt, hyp):
        '''
        :param ref: A list of strings with reference sentences
        :param hyp: A list of strings with hypothesis sentences
        :return: A list of list of XMS Scores [CLP_unigram, CLP_bigram, UMD_unigram, UMD_bigram, CLP_unigram_lm,...]
        '''

        src_preprocessed, hyp_preprocessed = self.preprocess(gt, hyp)
        scores = self.scorer.compute_xmoverscore(src_preprocessed, hyp_preprocessed,
                                                           ngram=self.n_gram, bs=self.bs, layer=self.layer)
        lm_perplexity = self.scorer.compute_perplexity(hyp_preprocessed, bs=1)

        return self.metric_combination(scores, lm_perplexity, [1, 0.1])

    def preprocess(self, src, hyp):
        s, t = self.lp.split('-')
        a = MosesDetokenizer(lang=s)
        src_detok = [a.detokenize(sent.split(' ')) for sent in src]
        b = MosesDetokenizer(lang=t)
        hyp_detok = [b.detokenize(sent.split(' ')) for sent in hyp]

        hyp_detok = [truecase.get_true_case(sent) for sent in hyp_detok]
        return src_detok, hyp_detok

    def metric_combination(self, a, b, alpha):
        return alpha[0] * np.array(a) + alpha[1] * np.array(b)

    def get_state(self):
        return metricState(name = self.name, version=self.version, language_models=self.model_name)


if __name__ == '__main__':
    b = XMoverScore(lp ="de-en")

    print(b(["Ein einfacher Satz zum Test."], ["A simple sentence for test."]))

