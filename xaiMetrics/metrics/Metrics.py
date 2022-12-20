class Metrics:
    def __init__(self, metrics = None):
        self.metrics = {}
        if not metrics or "BERTSCORE" in metrics:
            from xaiMetrics.metrics.wrappers.BertScore import BertScore
            self.metrics["BERTSCORE"] = BertScore()
