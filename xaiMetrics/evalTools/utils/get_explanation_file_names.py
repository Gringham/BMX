import os

# Helper to compute the filename for explanations using a path, dataset tag, lp, explainer and metric name
def get_explanation_file_names(base_path, dataset, lps, explainers, metrics):
    return [os.path.join(base_path, dataset + "__" + e + "__" + lp + "__" + m + ".dill") for m in list(metrics.keys())
            for e in list(explainers.keys()) for lp in lps]

def get_file_info(filename):
    return os.path.basename(filename)[:-5].split("__")
