import os
import types

import pandas as pd

from project_root import ROOT_DIR
from xaiMetrics.constants import REFERENCE_FREE, REFERENCE_BASED
from xaiMetrics.experiments.test_pmeans import generate_explanations_and_scores
from xaiMetrics.explainer.wrappers.Erasure import ErasureExplainer
from xaiMetrics.explainer.wrappers.Lime import LimeExplainer
from xaiMetrics.explainer.wrappers.Shap import ShapExplainer
from xaiMetrics.metrics.wrappers.BARTScore import BARTScore
from xaiMetrics.metrics.wrappers.BertScore import BertScore
from xaiMetrics.metrics.wrappers.Comet import CometQE
from xaiMetrics.metrics.wrappers.Transquest import TransQuest
from xaiMetrics.metrics.wrappers.XlmrCosineSim import XlmrCosineSim

if __name__ == '__main__':
    # Whether to generate new explanations using this configuration
    generate_explanations = False
    dataset_name = "realsumm"  # Tag to save explanations under

    # Language Pairs to evaluate on
    lps = ["en"]

    # Explainer to evaluate on
    explainers = {
        #'ErasureExplainer': ErasureExplainer(),
        'LimeExplainer': LimeExplainer(),
        #'ShapExplainer': ShapExplainer(),
    }

    if generate_explanations:
        # Metrics to evaluate on
        metrics = {
            'BERTSCORE_REF_BASED_ROBERTA_DEFAULT': BertScore(mode=REFERENCE_BASED),
            'BARTScore': BARTScore(mode=REFERENCE_BASED, batch_size=128),
            #'XLMRSBERT': XlmrCosineSim(mode=REFERENCE_BASED),
        }

    else:
        # Mock Objects
        metrics = {
            'BERTSCORE_REF_BASED_ROBERTA_DEFAULT': BertScore(mode=REFERENCE_BASED),
            'BARTScore': BARTScore(mode=REFERENCE_BASED, batch_size=128),
        }

        # Generate explanations and correlations


    def reader(lp):
        return pd.read_csv(
            os.path.join(ROOT_DIR, "xaiMetrics", "data", "REALSumm", "realsum_processed.tsv"),
            delimiter='\t')


    generate_explanations_and_scores(dataset_name,
                                     lps,
                                     explainers,
                                     metrics,
                                     reader,
                                     stratification=False,
                                     generate_explanations=False,
                                     correlation="kendall",
                                     system_level=True,
                                     only_scores=False)

    generate_explanations_and_scores(dataset_name,
                                     lps,
                                     explainers,
                                     metrics,
                                     reader,
                                     stratification=False,
                                     generate_explanations=False,
                                     correlation="pearson",
                                     system_level=False,
                                     appendix="-segment",
                                     only_scores=False)
