from xaiMetrics.constants import REFERENCE_FREE, REFERENCE_BASED
from xaiMetrics.metrics.utils.metricState import metricState
from bert_score import scorer
from importlib.metadata import version

from xaiMetrics.metrics.wrappers.MetricClass import MetricClass


class BertScore(MetricClass):
    '''
    A wrapper class for BERTScore from https://github.com/Tiiiger/bert_score by
    Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q. Weinberger, and Yoav Artzi. “BERTScore: Evaluating
    Text Generation with BERT”. In: International Conference on Learning Representations. 2020. url:
    https://openreview.net/forum?id=SkeHuCVFDr.
    '''

    name = 'BERTSCORE'

    def __init__(self, batch_size=64, lang='en', mode=REFERENCE_BASED, *args, **kwargs):
        self.mode = mode
        self.scorer = scorer.BERTScorer(*args, lang=lang, batch_size=batch_size, **kwargs)
        self.version = version("bert-score")

    def __call__(self, gt, hyp):
        '''
        Implementation from here, installed via pip: https://github.com/Tiiiger/bert_score
        :param gt: A list of strings with ground_truth sentences (reference or source)
        :param hyp: A list of strings with hypothesis sentences
        :return: A list of f1 values of BertScore
        '''
        return self.scorer.score(hyp, gt)[2].tolist()

    def get_state(self):
        return metricState(self.name, self.version, self.scorer.model_type)

if __name__ == '__main__':
    b = BertScore()
    print(b(["A test sentence"], ["A simple sentence for test"]))
