from rouge import Rouge
from xaiMetrics.constants import REFERENCE_BASED
from xaiMetrics.metrics.utils.metricState import metricState

from xaiMetrics.metrics.wrappers.MetricClass import MetricClass
from statistics import mean


class RougeMetric(MetricClass):
    '''
    '''

    name = 'ROUGE'

    def __init__(self, rouge_type="rouge-l", agg = "f"):
        self.mode = REFERENCE_BASED
        self.rouge_type = rouge_type
        self.version = "1.0.1"
        self.rouge = Rouge()
        self.agg = agg


    def __call__(self, gt, hyp):
        '''
        :param gt: A list of strings with ground_truth sentences (reference or source)
        :param hyp: A list of strings with hypothesis sentences
        :return: A list of f1 values of BertScore
        '''

        # Setup multi ref
        if not type(gt[0]) == list:
            scores = self.rouge.get_scores(hyp, gt, avg=False)
        else:
            scores = [self.rouge.get_scores([hyp[x]] * len(gt[x]), gt[x], avg=True) for x in range(len(hyp))]


        return [s[self.rouge_type][self.agg] for s in scores]

    def get_state(self):
        return metricState(self.name, self.version, self.scorer.model_type)

if __name__ == '__main__':
    b = RougeMetric()
    print(b(["A test sentence", "second"], ["Second", "second"]))
    print(b([["A test sentence", "ref2"], ["second", "erg"]], ["Second", "second"]))
