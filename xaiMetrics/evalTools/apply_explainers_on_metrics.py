import dill, os, time
import pandas as pd
from project_root import ROOT_DIR
from xaiMetrics.constants import REFERENCE_FREE
from xaiMetrics.explainer.Explainers import Explainers
from xaiMetrics.metrics.Metrics import Metrics
from xaiMetrics.metrics.wrappers.BertScore import BertScore


def apply_explainers_on_metrics(df, metrics=None, explainers=None, print_time=False, save_dir=None, dataset="UNDEFINED",
                                lp="UNDEFINED"):
    """
    :param df: A dataframe to get explanations for
    :param metrics: A dictionary of metricname - metric object to explain
    :param explainers: A dictionary of explainername - explainer object to explain with
    :param print_time: Prints computation time
    :param save_dir: If defined, explanations will be saved as dill files for each combination
    :param dataset: Name of the current dataset (important for naming the save file)
    :param lp: Name of the current language pair (important for naming the save file)
    :return:
    """
    if metrics == None:
        metrics = Metrics().metrics
    if explainers == None:
        explainers = Explainers().explainers

    attributions = {}

    for explainer_name, explainer in explainers.items():
        print("Applying explainer:", explainer_name)
        start_time = time.time()

        attributions[explainer_name] = explainer.explain_df(df=df, metrics=metrics)

        if save_dir:
            for metric_name in list(metrics.keys()):
                save_file = os.path.join(save_dir,
                                         dataset + "__" + explainer_name + "__" + lp + "__" + metric_name + ".dill")

                with open(save_file, 'wb') as pickle_file:
                    dill.dump(attributions[explainer_name][metric_name], pickle_file, -1)

        if print_time:
            print(explainer_name, 'Used Time', time.time() - start_time)

    return attributions


if __name__ == '__main__':
    df = pd.read_csv(os.path.join(ROOT_DIR, "xaiMetrics", "data", "eval4nlp21", "eval4nlp_dev_ro_en.tsv"),
                     delimiter='\t')

    # Example of using default
    attributions = apply_explainers_on_metrics(df,
                                               metrics={
                                                   'BERTSCORE_REF_FREE': BertScore(mode=REFERENCE_FREE)
                                               },
                                               print_time=True)
