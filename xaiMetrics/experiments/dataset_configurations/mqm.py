import os
import types

import pandas as pd

from project_root import ROOT_DIR
from xaiMetrics.constants import REFERENCE_FREE
from xaiMetrics.experiments.test_pmeans import generate_explanations_and_scores
from xaiMetrics.explainer.wrappers.Erasure import ErasureExplainer
from xaiMetrics.explainer.wrappers.Lime import LimeExplainer
#from xaiMetrics.explainer.wrappers.Shap import ShapExplainer
#from xaiMetrics.metrics.wrappers.BertScore import BertScore
#from xaiMetrics.metrics.wrappers.Comet import CometQE
#from xaiMetrics.metrics.wrappers.Transquest import TransQuest
#from xaiMetrics.metrics.wrappers.XlmrCosineSim import XlmrCosineSim

if __name__ == '__main__':
    # Whether to generate new explanations using this configuration
    generate_explanations = False
    dataset_name = "mqm21"  # Tag to save explanations under

    # Language Pairs to evaluate on
    lps = ["en-de", "zh-en"]

    # Explainer to evaluate on
    explainers = {
        #'ErasureExplainer': ErasureExplainer(),
        'LimeExplainer': LimeExplainer(),
        #'ShapExplainer': ShapExplainer(),
    }

    if generate_explanations:
        # Metrics to evaluate on
        # nli_model = "joeddav/xlm-roberta-large-xnli"
        nli_model = "C:\\Users\\Jirac\\.cache\\huggingface\\hub\\models--joeddav--xlm-roberta-large-xnli\\snapshots\\9c1619b90a142cd2913190d80d5f488d6612f57e",
        metrics = {
            #'COMET': CometQE(),
            #'Transquest': TransQuest(),
            #'BERTSCORE_REF_FREE_XNLI': BertScore(model_type=nli_model, mode=REFERENCE_FREE,
            #                                     num_layers=16),
            #'XLMRSBERT': XlmrCosineSim(),
        }

    else:
        # Mock Objects
        metrics = {
            'COMET': types.SimpleNamespace(),
            'Transquest': types.SimpleNamespace(),
            'BERTSCORE_REF_FREE_XNLI': types.SimpleNamespace(),
            'XLMRSBERT': types.SimpleNamespace(),
        }

        # Generate explanations and correlations


    def reader(lp):
        return pd.read_csv(
            os.path.join(ROOT_DIR, "xaiMetrics", "data", "MQM", "mqm21_{lp}.tsv".format(lp=lp)),
            delimiter='\t')


    # Currently Systemlevel is activated and onlyscores will lead to score files being created
    # Only scores = False will lead to correlation plots
    generate_explanations_and_scores(dataset_name,
                                     lps,
                                     explainers,
                                     metrics,
                                     reader,
                                     stratification=False,
                                     generate_explanations=generate_explanations,
                                     correlation="kendall",
                                     only_scores=False,
                                     system_level=True,
                                     appendix="-kendall-sys-level2",
                                     rm_ref=True)

