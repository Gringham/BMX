from xaiMetrics.constants import REFERENCE_FREE, REFERENCE_BASED
from xaiMetrics.metrics.utils.metricState import metricState
from bert_score import scorer
from importlib.metadata import version

from xaiMetrics.metrics.wrappers.MetricClass import MetricClass


class AnyMetric(MetricClass):
    '''
    A wrapper for any lambda function that is passed during the Object initialisation
    '''

    def __init__(self, metric_function, mode='ref_free', name="AnyMetric"):
        '''
        :param metric_function: A function that takes a ground truth sentence and a hypothesis sentence as input
        '''
        self.mode = mode
        self.metric_function = metric_function
        self.name = name

    def __call__(self, gt, hyp):
        '''
        :param gt: A list of strings with ground_truth sentences (reference or source)
        :param hyp: A list of strings with hypothesis sentences
        :return: A list of metric scores
        '''
        return self.metric_function(gt, hyp)

    def get_state(self):
        return metricState(self.name, "Undefined")

    def evaluate_df(self, df):
        if self.mode == REFERENCE_BASED:
            return self.__call__(df['REF'].tolist(), df['HYP'].tolist())
        elif self.mode == REFERENCE_FREE:
            return self.__call__(df['SRC'].tolist(), df['HYP'].tolist())


if __name__ == '__main__':
    print(AnyMetric(lambda gt, hyp: [len(g) + len(h) for g, h in zip(gt, hyp)])(["A test sentence", "Another one"],
                                                                                ["A simple sentence for test",
                                                                                 "a last one"]))
