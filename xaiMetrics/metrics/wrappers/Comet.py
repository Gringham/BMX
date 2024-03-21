from xaiMetrics.constants import REFERENCE_FREE, REFERENCE_BASED
from xaiMetrics.metrics.utils.metricState import metricState

from xaiMetrics.metrics.wrappers.MetricClass import MetricClass
from comet import download_model, load_from_checkpoint


class CometQE(MetricClass):
    name = 'CometQE'

    def __init__(self, mode=REFERENCE_FREE, *args, **kwargs):
        self.model_path = download_model("wmt21-comet-qe-mqm")
        self.model = load_from_checkpoint(self.model_path)
        self.mode = mode

    def __call__(self, gt, hyp):
        '''
        :param gt: A list of strings with ground_truth sentences (reference or source)
        :param hyp: A list of strings with hypothesis sentences
        :return: A list of scores
        '''

        data = [
            {
                "src": g,
                "mt": h,

            } for g, h in zip(gt, hyp)
        ]

        # Set num_workers to 0 on windows to prevent multiprocessing errors
        # https://discuss.pytorch.org/t/dataloader-multiprocessing-error-cant-pickle-odict-keys-objects-when-num-workers-0/43951/12
        model_output = self.model.predict(data, batch_size=32, gpus=1, num_workers=0)
        seg_scores, _ = model_output
        return seg_scores

    def get_state(self):
        return metricState(self.name, self.version, self.scorer.model_type)

if __name__ == '__main__':
    b = CometQE()
    print(sum(p.numel() for p in b.model.parameters()))
    print(b(["A test sentence","\"So Cummings was told that these units must be preserved in their entirety.\""], ["Satz1", "„Also wurde Cummings gesagt, dass diese Einheiten in ihrer Gesamtheit erhalten werden müssen\“"]))
