class Explainers:
    def __init__(self, explainers = None):
        self.explainers = {}
        if not explainers or "ErasureExplainer" in explainers:
            from xaiMetrics.explainer.wrappers.Erasure import ErasureExplainer
            self.explainers["ErasureExplainer"] = ErasureExplainer()
