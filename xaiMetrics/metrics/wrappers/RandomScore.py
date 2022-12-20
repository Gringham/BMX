import random

import sacrebleu
from sacremoses import MosesTokenizer

from xaiMetrics.constants import REFERENCE_BASED, REFERENCE_FREE
from xaiMetrics.metrics.utils.metricState import metricState
from xaiMetrics.metrics.wrappers.MetricClass import MetricClass
from importlib.metadata import version

from easynmt import EasyNMT


class RandomScore(MetricClass):
    name = 'RandomScore'

    def __init__(self, mode=REFERENCE_BASED, hyp_lang = 'en', easyNMTModel = 'm2m_100_1.2B', bs = 8, lp = None):
        self.version = "0"
        self.mode=REFERENCE_FREE

    def __call__(self, gt, hyp):
        return [random.uniform(0, 1) for g in gt]

    def get_state(self):
        return metricState(self.name, self.version)

    def evaluate_df(self, df):
        src = df['SRC'].tolist()
        return self.__call__(src, src)




if __name__ == '__main__':
    b = RandomScore()

    # Sample using ref and hyp lists
    print(b(["A simple  for test"],["A simple sentence for test"]))
    #[0.44721359549995787]

    print(b.get_state())

    b = RandomScore()

    # Sample using ref and hyp lists
    print(b(["Ein leichter Test"], ["A simple sentence for test"]))


    print(b.get_state())