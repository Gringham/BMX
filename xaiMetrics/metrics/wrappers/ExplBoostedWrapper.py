import pandas as pd

from xaiMetrics.constants import REFERENCE_FREE
from xaiMetrics.evalTools.apply_explainers_on_metrics import apply_explainers_on_metrics
from xaiMetrics.explainer.wrappers.Erasure import ErasureExplainer
from xaiMetrics.explanations.FeatureImportanceExplanation import FeatureImportanceExplanationCollection
from xaiMetrics.metrics.utils.metricState import metricState
from xaiMetrics.metrics.wrappers.BertScore import BertScore
from xaiMetrics.metrics.wrappers.MetricClass import MetricClass



class ExplBoostedWrapper(MetricClass):
    '''
    A wrapper of the explainability boosted metrics that are the cores of our experiments. This wrapper can be used
    in order to realize further iteration of our approach. For multiple iterations, it has to be implemented in a
    recursvie loop.
    '''
    def __init__(self, metric, explainer, p, w, mode=REFERENCE_FREE):
        self.version = 0
        self.name = 'ExplBoostedWrapper'
        self.metric = metric
        self.explainer = explainer
        self.p = p
        self.w = w
        self.mode = mode


    def __call__(self, gt, hyp, called = False):
        if self.mode == REFERENCE_FREE:
            df = pd.DataFrame([[g,h] for g,h in zip(gt, hyp)], columns=["SRC", "HYP"])
        else:
            df = pd.DataFrame([[g, h] for g, h in zip(gt, hyp)], columns=["REF", "HYP"])
        attributions = apply_explainers_on_metrics(df,
                           explainers={"E":self.explainer},
                           metrics={"M":self.metric})
        importance = FeatureImportanceExplanationCollection(attributions["E"]["M"])
        e = importance.combine_pmean_with_scores([self.w], [self.p], only_hyp=False)
        return [a[0] for a in e[0].tolist()[0]]

    def get_state(self):
        return metricState(self.name, self.version)

    def evaluate_df(self, df):
        return self.__call__(df['REF'].tolist(), df['HYP'].tolist(), called=True)




if __name__ == '__main__':
    b = ExplBoostedWrapper(explainer=ErasureExplainer(),
                           metric=BertScore(model_type="joeddav/xlm-roberta-large-xnli",
                                            mode=REFERENCE_FREE,
                                            num_layers=16),
                           p=0.6,
                           w=-0.3
                           )

    # Sample using ref and hyp lists
    print(b(["A simple  for test", "A second simple test"],["A simple sentence for test", "Just one more test"]))
    #[0.44721359549995787]


