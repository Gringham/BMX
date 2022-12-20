import sacrebleu
from sacremoses import MosesTokenizer

from xaiMetrics.constants import REFERENCE_BASED, REFERENCE_FREE
from xaiMetrics.metrics.utils.metricState import metricState
from xaiMetrics.metrics.wrappers.MetricClass import MetricClass
from importlib.metadata import version

from easynmt import EasyNMT


class SentenceBleu(MetricClass):
    '''A wrapper for SacreBleu SentenceBleu (https://github.com/mjpost/sacrebleu) a Sentence BLEU implementation by:
    Matt Post. “A Call for Clarity in Reporting BLEU Scores”. In: Proceedings of the Third Conference on
    Machine Translation: Research Papers. Brussels, Belgium: Association for Computational Linguistics, Oct.
    2018, pp. 186–191. doi: 10.18653/v1/W18-6319. url: https://aclanthology.org/W18-
    6319.
    '''
    name = 'SENTENCEBLEU'


    def __init__(self, mode=REFERENCE_BASED, hyp_lang = 'en', easyNMTModel = 'm2m_100_1.2B', bs = 8, lp = None):
        self.tokenizer = MosesTokenizer(lang=hyp_lang)
        self.version = version("sacrebleu")
        self.mode = mode
        self.lp = lp

        if self.mode == REFERENCE_FREE:
            self.model = EasyNMT(easyNMTModel)
            self.bs = bs

    def __call__(self, gt, hyp, called = False):
        '''
        ! Warning, reference free mode can only be called from "evaluate_df" function due to performance considerations
        :param ref: A list of strings with reference sentences
        :param hyp: A list of strings with hypothesis sentences
        :return: A list of SacreBleu Sentence Bleu scores per reference - hypothesis pair
        '''
        if self.mode == REFERENCE_BASED or called:
            return [sacrebleu.sentence_bleu(hyp[x], [gt[x]], smooth_method='add-k', smooth_value=1).score/100 for x in range(len(hyp))]
        s, h = self.lp.split('-')
        src_transl = self.model.translate(gt, source_lang=s, target_lang=h)
        return [sacrebleu.sentence_bleu(hyp[x], [src_transl[x]], smooth_method='add-k', smooth_value=1).score / 100 for x
                in range(len(hyp))]

    def get_state(self):
        return metricState(self.name, self.version)

    def evaluate_df(self, df):
        if self.mode == REFERENCE_BASED:
            return self.__call__(df['REF'].tolist(), df['HYP'].tolist(), called=True)

        src = df['SRC'].tolist()
        lp = set(df['LP'].tolist())
        if not len(lp) == 1:
            raise Exception("Translation Bleu Expects Dataframe with uniform language pair (for now)")

        s, h = lp[0].split('-')
        src_transl = self.model.translate(src, source_lang=s, target_lang=h)
        return self.__call__(src_transl, df['HYP'].tolist(), called=True)




if __name__ == '__main__':
    b = SentenceBleu(mode=REFERENCE_BASED)

    # Sample using ref and hyp lists
    print(b(["A simple  for test"],["A simple sentence for test"]))
    #[0.44721359549995787]

    print(b.get_state())

    b = SentenceBleu(mode=REFERENCE_FREE, lp="de-en")

    # Sample using ref and hyp lists
    print(b(["Ein leichter Test"], ["A simple sentence for test"]))


    print(b.get_state())