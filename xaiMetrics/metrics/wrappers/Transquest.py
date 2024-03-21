import torch
from transquest.algo.sentence_level.monotransquest.run_model import MonoTransQuestModel


from xaiMetrics.constants import REFERENCE_FREE, REFERENCE_BASED
from xaiMetrics.metrics.utils.metricState import metricState

from xaiMetrics.metrics.wrappers.MetricClass import MetricClass


class TransQuest(MetricClass):
    name = 'TransQuest'

    def __init__(self, mode=REFERENCE_FREE, *args, **kwargs):
        self.mode = mode
        self.scorer = MonoTransQuestModel("xlmroberta", "TransQuest/monotransquest-da-multilingual", num_labels=1, use_cuda=torch.cuda.is_available())

    def __call__(self, gt, hyp):
        '''
        :param gt: A list of strings with ground_truth sentences (reference or source)
        :param hyp: A list of strings with hypothesis sentences
        :return: A list of f1 values of BertScore
        '''
        res = self.scorer.predict([[gt[x],hyp[x]] for x in range(len(gt))])[0].tolist()
        if type(res) == float:
            res = [res]
        return res

    def get_state(self):
        return metricState(self.name, self.version, self.scorer.model_type)

if __name__ == '__main__':
    b = TransQuest()
    print(sum(p.numel() for p in b.scorer.model.parameters()))
    print(b(["A test sentence","\"So Cummings was told that these units must be preserved in their entirety.\"", ""], ["Satz1", "„Also wurde Cummings gesagt, dass diese Einheiten in ihrer Gesamtheit erhalten werden müssen\“", ""]))
