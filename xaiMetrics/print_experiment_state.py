# Function to iterate over all Metric and Explainer States to output installed library versions
from xaiMetrics.constants import REFERENCE_FREE, REFERENCE_BASED
from xaiMetrics.explainer.wrappers.Lime import LimeExplainer
from xaiMetrics.explainer.wrappers.Shap import ShapExplainer
from xaiMetrics.metrics.wrappers.BertScore import BertScore
from xaiMetrics.metrics.wrappers.SentenceBleu import SentenceBleu
from xaiMetrics.metrics.wrappers.XMoverScore import XMoverScore
from xaiMetrics.metrics.wrappers.XlmrCosineSim import XlmrCosineSim
from importlib.metadata import version

explainers = {
    'LimeExplainer': LimeExplainer(),
    'ShapExplainer': ShapExplainer(),
}

metrics = {
    'BERTSCORE_REF_FREE_XLMR': BertScore(model_type="xlm-roberta-large", mode=REFERENCE_FREE),
    'BERTSCORE_REF_BASED_ROBERTA_DEFAULT': BertScore(mode=REFERENCE_BASED),
    'BERTSCORE_REF_FREE_XNLI': BertScore(model_type="joeddav/xlm-roberta-large-xnli", mode=REFERENCE_FREE,
                                        num_layers=16),
    'SentenceBLEU_REF_BASED': SentenceBleu(mode=REFERENCE_BASED),
    'XLMRSBERT': XlmrCosineSim(),
    'XMoverScore_No_Mapping': XMoverScore(),
}

print("torch:", version("torch"))
print("transformers:", version("transformers"))
