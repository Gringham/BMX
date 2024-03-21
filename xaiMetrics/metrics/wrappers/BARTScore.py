import os

from project_root import ROOT_DIR
from xaiMetrics.constants import REFERENCE_BASED
from xaiMetrics.metrics.utils.metricState import metricState
from importlib.metadata import version

from xaiMetrics.metrics.wrappers.MetricClass import MetricClass
from xaiMetrics.metrics.wrappers.source.BARTScore.bart_score import BARTScorer


class BARTScore(MetricClass):
    '''
    '''

    name = 'BARTSCORE'

    def __init__(self, batch_size=8, lang='en', mode=REFERENCE_BASED, *args, **kwargs):
        self.mode = mode
        self.bart_scorer = BARTScorer(device='cuda:0', checkpoint='facebook/bart-large-cnn')
        self.bart_scorer.load(path=os.path.join(ROOT_DIR, "xaiMetrics", "metrics", "wrappers", "models", "bart_score.pth"))
        self.version = "05-22"
        self.batch_size = batch_size


    def __call__(self, gt, hyp):
        '''
        '''
        if type(gt[0]) == list:
            return self.bart_scorer.multi_ref_score(hyp, gt, batch_size=self.batch_size)
        else:
            return self.bart_scorer.score(hyp, gt, batch_size=self.batch_size)


    def get_state(self):
        return metricState(self.name, self.version, "bart")

if __name__ == '__main__':
    b = BARTScore()

    print(sum(p.numel() for p in b.bart_scorer.model.parameters()))
    print(b([["A test sentence"]],['"So Cummings was told that these units must be preserved in their entirety."']))
