import os
from statistics import mean, median

import pytorch_lightning as pl
from more_itertools import chunked
from sklearn.linear_model import LogisticRegression, LinearRegression

from transformers import AutoTokenizer
from transformers import AutoModel

import torch

from project_root import ROOT_DIR
from xaiMetrics.constants import REFERENCE_FREE
from xaiMetrics.metrics.utils.metricState import metricState
from xaiMetrics.metrics.wrappers.MetricClass import MetricClass
import pandas as pd

class XlmrCosineSimLG(MetricClass, pl.LightningModule):
    '''A wrapper class for XLMR sentence embedding cosine similarity. I base on the implementation in:
        https://huggingface.co/sentence-transformers/LaBSE/blob/f84a947c7a83d42cdf9b1a87ca014dbeb0807d06/README.md
        XLMR is proposed by:
        Alexis Conneau and Guillaume Lample. “Cross-lingual Language Model Pretraining”. In: Advances in
        Neural Information Processing Systems. Ed. by H. Wallach, H. Larochelle, A. Beygelzimer, F. d’Alché-
        Buc, E. Fox, and R. Garnett. Vol. 32. Curran Associates, Inc., 2019.
        url: https://proceedings.neurips.cc/paper/2019/file/c04c19c2c2474dbf5f7ac4372c5b9af1-Paper.pdf.
        Sentence Transformers are provided by:
        Nils Reimers and Iryna Gurevych. “Making Monolingual Sentence Embeddings Multilingual using
        Knowledge Distillation”. In: Proceedings of the 2020 Conference on Empirical Methods in Natural
        Language Processing. Association for Computational Linguistics, Nov. 2020.
        url: https://arxiv.org/abs/2004.09813.'''
    name = 'XLMRCosineSimSentenseTransformers'


    def __init__(self, bs=16, mode=REFERENCE_FREE):
        super().__init__()
        # following the setup from here: https://huggingface.co/sentence-transformers/stsb-xlm-r-multilingual
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/stsb-xlm-r-multilingual',
                                                       add_special_tokens=False)

        # Loading the model 2 times in order to be able to use integrated gradients to manipulate its embeddings
        self.model = AutoModel.from_pretrained('sentence-transformers/stsb-xlm-r-multilingual')
        self.cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        self.bs = bs
        self.version = "custom"
        self.mode = mode
        self.lr_coherence = LinearRegression()
        self.lr_consistency = LinearRegression()
        self.lr_fluency = LinearRegression()
        self.lr_relevance = LinearRegression()
        self.fitted = False

    def __call__(self, src, hyp):
        '''
        :param ref: A list of strings with source sentences
        :param hyp: A list of strings with hypothesis sentences
        :return: A list of Labse Scores
        '''
        if type(src[0]) == list:
            src = [s[0] for s in src]
        self.eval()

        src_ids = self.tokenize(src)
        hyp_ids = self.tokenize(hyp)

        self.to('cuda:0')
        scores_coherence = []
        scores_consistency = []
        scores_fluency = []
        scores_relevance = []

        joint = []

        # Unfortunately I didn't find existing batching here (so I'd suppose it should be there)
        # Therefore I use some other lib for batching
        for x in chunked(range(src_ids['input_ids'].shape[0]), self.bs):
            s_in_batch = src_ids['input_ids'][x].cuda()
            h_in_batch = hyp_ids['input_ids'][x].cuda()
            s_att_batch = src_ids['attention_mask'][x].cuda()
            h_att_batch = hyp_ids['attention_mask'][x].cuda()

            src_emb, hyp_emb = self.forward(s_in_batch, h_in_batch,
                               s_att_batch, h_att_batch)
            ct = torch.cat([src_emb, hyp_emb], dim=1)
            for c in ct:
                joint.append(c.detach().cpu().numpy())
                if self.fitted == True:
                    scores_coherence.append(self.lr_coherence.predict([joint[-1]])[0])
                    scores_consistency.append(self.lr_consistency.predict([joint[-1]])[0])
                    scores_fluency.append(self.lr_fluency.predict([joint[-1]])[0])
                    scores_relevance.append(self.lr_relevance.predict([joint[-1]])[0])
                else:
                    scores_coherence.append(0)
                    scores_consistency.append(0)
                    scores_fluency.append(0)
                    scores_relevance.append(0)

        return scores_coherence, scores_consistency, scores_fluency, scores_relevance, joint

    def tokenize(self, sent):
        return self.tokenizer(sent, padding=True, truncation=True, max_length=128, return_tensors='pt')

    # Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(self, model_output, attention_mask):
        # pooling step from here: https://huggingface.co/sentence-transformers/stsb-xlm-r-multilingual
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, src_ids, hyp_ids, src_mask, hyp_mask):
        src_model_out = self.model(input_ids=src_ids, attention_mask=src_mask)
        src_emb = self.mean_pooling(src_model_out, src_mask)

        hyp_model_out = self.model(input_ids=hyp_ids, attention_mask=hyp_mask)
        hyp_emb = self.mean_pooling(hyp_model_out, hyp_mask)

        return src_emb, hyp_emb


    def fit_regression(self):
        summ_df = pd.read_json(
            os.path.join(ROOT_DIR, "xaiMetrics", "data", "cnndm", "SummEval.json"))
        summ_df = summ_df.head(208)
        summ_df["DA_Coherence"] = [d["coherence"] for d in summ_df["expert_avg"].to_list()]
        summ_df["DA_Consistency"] = [d["consistency"] for d in summ_df["expert_avg"].to_list()]
        summ_df["DA_Fluency"] = [d["fluency"] for d in summ_df["expert_avg"].to_list()]
        summ_df["DA_Relevance"] = [d["relevance"] for d in summ_df["expert_avg"].to_list()]

        summ_df = summ_df.sample(frac=1).reset_index(drop=True)

        ref = sum([r for r in summ_df["REF"].tolist()], [])
        hyp = sum([[h] * len(summ_df["REF"].tolist()[0]) for h in summ_df["HYP"].tolist()], [])
        da_coherence = sum([[d] * len(summ_df["REF"].tolist()[0]) for d in summ_df["DA_Coherence"].tolist()], [])
        da_consistency = sum([[d] * len(summ_df["REF"].tolist()[0]) for d in summ_df["DA_Consistency"].tolist()], [])
        da_fluency = sum([[d] * len(summ_df["REF"].tolist()[0]) for d in summ_df["DA_Fluency"].tolist()], [])
        da_relevance = sum([[d] * len(summ_df["REF"].tolist()[0]) for d in summ_df["DA_Relevance"].tolist()], [])

        _,_,_,_, emb = self.__call__(ref, hyp)
        self.lr_coherence.fit(emb, da_coherence)
        self.lr_consistency.fit(emb, da_consistency)
        self.lr_fluency.fit(emb, da_fluency)
        self.lr_relevance.fit(emb, da_relevance)
        self.fitted = True

    def get_state(self):
        return metricState(self.name, self.version, self.hyp_model)


if __name__ == '__main__':
    c = XlmrCosineSimLG()
    c.fit_regression()


    # Sample using ref and hyp lists
    # [0.9784649610519409]
    print(c(["Ein Test Satz"], ["A test sentence"]))


