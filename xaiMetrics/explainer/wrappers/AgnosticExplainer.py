import abc

import tqdm

from xaiMetrics.constants import REFERENCE_FREE


class AgnosticExplainer(metaclass=abc.ABCMeta):
    '''
    This class is an abstract class for agnostic explainers
    '''

    @abc.abstractmethod
    def explain_sentence(self, gt, hyp, metric):
        raise NotImplementedError

    def explain_df(self, df, metrics):
        attributions = {}
        for name, metric in metrics.items():
            tag = "Explanations" + name + ": "
            hyp = df['HYP'].tolist()
            if metric.mode == REFERENCE_FREE:
                gt = df['SRC'].tolist()
            else:
                gt = df['REF'].tolist()
            attributions[name] = [self.explain_sentence(gt, hyp, metric) for gt, hyp in
                                  tqdm.tqdm(zip(gt, hyp), desc=tag)]

        return attributions
