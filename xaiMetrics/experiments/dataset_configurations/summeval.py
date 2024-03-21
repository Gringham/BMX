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
    dataset_name = "SummEval"  # Tag to save explanations under

    # Language Pairs to evaluate on
    lps = ["en"]

    # Explainer to evaluate on
    explainers = {
        # 'ErasureExplainer': ErasureExplainer(),
        'LimeExplainer': LimeExplainer(),
        # 'ShapExplainer': ShapExplainer(),
    }

    if generate_explanations:
        # Metrics to evaluate on
        # nli_model = "joeddav/xlm-roberta-large-xnli"
        nli_model = "C:\\Users\\Jirac\\.cache\\huggingface\\hub\\models--joeddav--xlm-roberta-large-xnli\\snapshots\\9c1619b90a142cd2913190d80d5f488d6612f57e",
        metrics = {
            'BERTSCORE_REF_BASED_ROBERTA_DEFAULT': BertScore(mode=REFERENCE_BASED),
            'BARTScore': BARTScore(mode=REFERENCE_BASED, batch_size=128),
            'XLMRSBERT': XlmrCosineSim(mode=REFERENCE_BASED),
        }

    else:
        # Mock Objects
        metrics = {
            'BERTSCORE_REF_BASED_ROBERTA_DEFAULT': types.SimpleNamespace(),
            'BARTScore': types.SimpleNamespace(),
        }

        # Generate explanations and correlations

    for property in ["coherence", "consistency", "fluency", "relevance"]:
        def reader(lp):
            df = pd.read_json(
                os.path.join(ROOT_DIR, "xaiMetrics", "data", "cnndm", "SummEval.json"))
            df["DA"] = [d[property] for d in df["expert_avg"].to_list()]
            return df


        for correlation in ["kendall", "spearman"]:
            #generate_explanations_and_scores(dataset_name,
            #                                 lps,
            #                                 explainers,
            #                                 metrics,
            #                                 reader,
            #                                 stratification=True,
            #                                 generate_explanations=generate_explanations,
            #                                 correlation=correlation,
            #                                 system_level=True,
            #                                 first_ref=False,
            #                                 appendix=correlation + "-" + property + "-first-ref")

            generate_explanations_and_scores(dataset_name,
                                             lps,
                                             explainers,
                                             metrics,
                                             reader,
                                             stratification=False,
                                             generate_explanations=generate_explanations,
                                             correlation=correlation,
                                             system_level=True,
                                             first_ref=False,
                                             only_scores=False,
                                             appendix="SummEval-complete"+property+"-"+correlation)
