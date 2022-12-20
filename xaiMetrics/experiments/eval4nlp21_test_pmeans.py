import tqdm

from project_root import ROOT_DIR

import pandas as pd
import os, dill

from xaiMetrics.constants import REFERENCE_FREE, REFERENCE_BASED
from xaiMetrics.evalTools.apply_explainers_on_metrics import apply_explainers_on_metrics
from xaiMetrics.evalTools.explanations_to_scores import explanations_to_scores_and_eval
from xaiMetrics.evalTools.utils.get_explanation_file_names import get_explanation_file_names, get_file_info
from xaiMetrics.explainer.wrappers.Erasure import ErasureExplainer
from xaiMetrics.explainer.wrappers.InputMarginalization import InputMarginalizationExplainer
from xaiMetrics.explainer.wrappers.Lime import LimeExplainer
from xaiMetrics.explainer.wrappers.RandomExplainer import RandomExplainer
from xaiMetrics.explainer.wrappers.Shap import ShapExplainer
from xaiMetrics.explanations.FeatureImportanceExplanation import FeatureImportanceExplanationCollection
from xaiMetrics.metrics.wrappers.BertScore import BertScore
from xaiMetrics.metrics.wrappers.RandomScore import RandomScore
from xaiMetrics.metrics.wrappers.SentenceBleu import SentenceBleu
from xaiMetrics.metrics.wrappers.XMoverScore import XMoverScore
from xaiMetrics.metrics.wrappers.XlmrCosineSim import XlmrCosineSim


def generate_explanations_and_scores():
    """
    This method creates new explanations for eval4nlp, applies the powermean and calculates the new correlations
    """

    # Language Pairs to evaluate on
    lps = ["ro-en", "et-en", "ru-de", "de-zh"]

    # Explainer to evaluate on
    explainers = {
        'ErasureExplainer': ErasureExplainer(),
        'LimeExplainer': LimeExplainer(),
        'ShapExplainer': ShapExplainer(),
        'InputMarginalizationExplainer': InputMarginalizationExplainer(delta=0.05),
        'RandomExplainer': RandomExplainer()
    }

    # Metrics to evaluate on
    metrics = {
        'BERTSCORE_REF_BASED_ROBERTA_DEFAULT': BertScore(mode=REFERENCE_BASED),
        'BERTSCORE_REF_FREE_XNLI': BertScore(model_type="joeddav/xlm-roberta-large-xnli", mode=REFERENCE_FREE,
                                             num_layers=16),
        'SentenceBLEU_REF_BASED': SentenceBleu(mode=REFERENCE_BASED),
        'XLMRSBERT': XlmrCosineSim(),
        'XMoverScore_No_Mapping': XMoverScore(),
        'RandomScore': RandomScore()
    }


    dataset = "eval4_nlp_21_test" # Tag to save explanations under
    generate_explanations = True # Set to false, if existing explanations should be used

    # Where to save explanations. Filenames are pregenerated
    explanationDirectory = os.path.join(ROOT_DIR, "xaiMetrics", "outputs", "raw_explanations")
    explanationFilenames = get_explanation_file_names(explanationDirectory, dataset, lps, explainers, metrics)
    correlations = []

    for lp in tqdm.tqdm(lps, desc="LP:"):

        for metric in metrics:
            metrics[metric].lp = lp

        print("Explaining:", lp)

        # Read in pandas corpus
        df = pd.read_csv(
            os.path.join(ROOT_DIR, "xaiMetrics", "data", "eval4nlp21", "eval4nlp_{set}_{lp}.tsv".format(lp=lp, set="test")),
            delimiter='\t')

        if generate_explanations:
            # Generates explanations for all configured explainers and metrics
            print("Applying Explainers and metrics")
            attributions = apply_explainers_on_metrics(df,
                                                       explainers=explainers,
                                                       metrics=metrics,
                                                       print_time=True,
                                                       save_dir=explanationDirectory,
						       dataset=dataset,
                                                       lp=lp)

        # combining with pmeans
        attributions = {}
        for filename in explanationFilenames:
            _, explainer, file_lp, metric = get_file_info(filename)
            if lp == file_lp:
                with open(filename, 'rb') as pickle_file:
                    print("Reading generated explanations from file: ", pickle_file)
                    if explainer not in attributions:
                        attributions[explainer] = {}
                    attributions[explainer][metric] = FeatureImportanceExplanationCollection(dill.load(pickle_file))

        correlations.append(explanations_to_scores_and_eval(df, attributions, dataset=dataset, lp=lp, save_fig=True, only_hyp = False))

    df = pd.concat(correlations)
    print(df.to_string())
    print(df.groupby([df.metric, df.explainer]).mean().to_string())

if __name__ == '__main__':
    generate_explanations_and_scores()
