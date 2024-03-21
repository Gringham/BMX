import time
from transformers import BertTokenizer, BertForMaskedLM
import torch
import numpy as np

from xaiMetrics.constants import REFERENCE_FREE
from xaiMetrics.explainer.wrappers.AgnosticExplainer import AgnosticExplainer
from xaiMetrics.explanations.FeatureImportanceExplanation import FeatureImportanceExplanation
from xaiMetrics.metrics.wrappers.AnyMetric import AnyMetric
from xaiMetrics.metrics.wrappers.BertScore import BertScore


class InputMarginalizationExplainer(AgnosticExplainer):
    '''
    Implementation of Input Marginalization by

    Siwon Kim, Jihun Yi, Eunji Kim, and Sungroh Yoon. “Interpretation of NLP models through input
    marginalization”. In: Proceedings of the 2020 Conference on Empirical Methods in Natural Language
    Processing (EMNLP). Online: Association for Computational Linguistics, Nov. 2020, pp. 3154–3167. doi:
    10.18653/v1/2020.emnlp-main.255. url: https://aclanthology.org/2020.emnlp-main.255.

    for regression.

    The class iterates each token and for each token evaluates which effect its replacement with likely other tokens
    has on a metric.
    '''

    def __init__(self, delta=0.0001):
        # Initialize a model for the replacement
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.vocab = list(self.tokenizer.ids_to_tokens.values())
        self.model = BertForMaskedLM.from_pretrained('bert-base-multilingual-cased')

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.delta = delta

    def weight_of_evidence(self):
        # Not used here, as we don't have probabilities
        pass

    def explain_sentence(self, gt, hyp, metric):
        '''
        :param hyp: A hypothesis sentence
        :param metric: A metric with fixed source or reference
        :param delta: A minimal probability a token should have based on self.model to be replaced with
        :return: (a list of attribution tuples (importance, token) ; the original score)
        '''
        s = time.time()
        hyp_tokens = self.collapse_list(self.tokenizer.tokenize(hyp))

        if len(hyp_tokens) > 80:
            hyp_tokens = hyp_tokens[:80]
            print("Cutting hypothesis sentence after 80 tokens to handle it performancewise")
        hyp_probab = self.get_probabilities(hyp_tokens, self.delta)
        hyp_sents, hyp_indices = self.permute_sents(hyp_tokens, hyp_probab, hyp)

        gt_tokens = self.collapse_list(self.tokenizer.tokenize(gt))
        gt_probab = self.get_probabilities(gt_tokens, self.delta)
        gt_sents, gt_indices = self.permute_sents(gt_tokens, gt_probab, gt)

        # get the original score and permuted scores
        real, gt_scores, hyp_scores = self.get_scores(gt_sents, hyp_sents, gt_indices, hyp_indices, gt_probab, hyp_probab, metric)

        # flatten the list and normalize the probabilities using softmax
        hyp_probs = [[hyp_probab[y][x][0] for x in range(len(hyp_probab[y]))] for y in range(len(hyp_probab))]
        hyp_probs = [np.exp(p) / sum(np.exp(p)) for p in hyp_probs]

        gt_probs = [[gt_probab[y][x][0] for x in range(len(gt_probab[y]))] for y in range(len(gt_probab))]
        gt_probs = [np.exp(p) / sum(np.exp(p)) for p in gt_probs]

        # weight the scores by their probabilities
        hyp_weighted_scores = [[hyp_probs[y][x] * hyp_scores[y][x] for x in range(len(hyp_probs[y]))] for y in range(len(hyp_probs))]
        hyp_probable_scores = [sum(w) for w in hyp_weighted_scores]

        gt_weighted_scores = [[gt_probs[y][x] * gt_scores[y][x] for x in range(len(gt_probs[y]))] for y in range(len(gt_probs))]
        gt_probable_scores = [sum(w) for w in gt_weighted_scores]

        # get probabilities as the difference between the original and the replacements
        hyp_attributions = [(z[0], z[1]) for z in list(zip([real - p for p in hyp_probable_scores], hyp_tokens))]
        gt_attributions = [(z[0], z[1]) for z in list(zip([real - p for p in gt_probable_scores], gt_tokens))]

        return FeatureImportanceExplanation(real, gt, hyp, gt_attributions, hyp_attributions, mode=metric.mode)

    def permute_sents(self, tokens, probab, hyp):
        '''
        :param tokens: A list of tokenized input sentences
        :param probab: A list of probability, token pairs for each sentence [[(0.5367058372591685, 'the'), (0.25725506801287173, 'my')], [(0.22504414071595363, 'mother'), ...
        :param hyp: The original hypothesis
        :return: A list of perturbed sentences, Indices for the position in probab. Using these allows for efficient metric evaluation
        '''
        sents = []
        indices = []
        for x in range(len(probab)):
            for y in range(len(probab[x])):
                s = tokens.copy()
                s[x] = probab[x][y][1]
                sents.append(s)
                indices.append(x)

        sents = [' '.join(s) for s in sents]

        # Adding the real sentence to the end
        sents.append(hyp)
        return sents, indices

    def get_probab(self, hyp, delta):
        '''
        Precomputes the permuted sentences and returns a dictionary
        :param hyp: A list of input sentence
        :param delta: A minimal probability for values
        :return:
        '''
        hyp_tokens = self.collapse_list(self.tokenizer.tokenize(hyp))
        probab = self.get_probabilities(hyp_tokens, delta)
        print('Produced', sum(len(l) for l in probab), 'close words')

        sents, indices = self.permute_sents(hyp_tokens, probab, hyp)
        return {'pre_tokens': hyp_tokens, 'pre_probab': probab, 'pre_sents': sents, 'pre_indices': indices}

    def get_scores(self, gt_sents, hyp_sents, gt_indices, hyp_indices, gt_probab, hyp_probab, metric):
        '''
        Works with the output of permute sents
        :param sents: permuted sentences
        :param indices: The index of the position that was permuted
        :param probab: This list contains a tuple of probability score and word for every word assigned a score higher than delta
        :param metric: A metric to explain
        :return:
        '''
        scores = metric(gt_sents[:-1]+[gt_sents[-1]]*len(hyp_sents)+[gt_sents[-1]], [hyp_sents[-1]]*len(gt_sents)+hyp_sents)

        # group metric scores by index
        gt_res = [[scores[:len(gt_sents)][y] for y in range(len(gt_indices)) if gt_indices[y] == x] for x in range(len(gt_probab))]
        hyp_res = [[scores[len(gt_sents):][y] for y in range(len(hyp_indices)) if hyp_indices[y] == x] for x in range(len(hyp_probab))]

        # A list of lists with prediction results for the sentences with replaced words. Also adding the calculation of
        # the real score here
        return scores[-1], gt_res, hyp_res

    def get_probabilities(self, tokens, delta=0.0001):
        # Returns a list of word probabilities higher than delta for every masked out token
        # [[(0.5367058372591685, 'the'), (0.25725506801287173, 'my')], [(0.22504414071595363, 'mother'), (0.20174577159349558, 'mom')], ...
        s = time.time()
        # Generates a list with lists of tokens for masking out every possible word in a sentence and add the cls and sep
        tokens = [[tokens[x] if x != y else '[MASK]' for x in range(len(tokens))] for y in range(len(tokens))]
        tokens = [self.tokenizer.tokenize(' '.join(['[CLS]'] + t + ['[SEP]'])) for t in tokens]

        # Generates segment ids based on the token lengths an determines max len for padding
        max_len = max(len(t) for t in tokens)
        seg_ids = [[0] * len(i) for i in tokens]

        # Generating the padding with a different id
        for x in range(len(tokens)):
            while len(tokens[x]) < max_len:
                tokens[x].append('[PAD]')
                seg_ids[x].append(1)

        # Getting the index after re-tokenization
        mask_indices = [t.index('[MASK]') for t in tokens]

        # Getting the ids for every token
        ids = [self.tokenizer.convert_tokens_to_ids(t) for t in tokens]

        # Move ids to tensor
        id_tensor = torch.tensor(ids, device=self.device)
        seg_id_tensor = torch.tensor(seg_ids, device=self.device)

        # Get Predictions
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(id_tensor, seg_id_tensor)

        # Get the predictions for the masked token
        bert_scores = [predictions[0][x, mask_indices[x]].tolist() for x in range(predictions[0].shape[0])]

        # Use softmax to obtain probabilities
        bert_scores_soft = [np.exp(b) / sum(np.exp(b)) for b in bert_scores]

        # Order by tokens with the highest score and filter by delta
        scoring = [list(zip(bert_scores_soft[x], self.vocab)) for x in range(len(bert_scores_soft))]
        scoring = [[w for w in s if w[0] > delta] for s in scoring]

        [s.sort(key=lambda x: x[0], reverse=True) for s in scoring]

        return scoring

    def collapse_list(self, l):
        '''
        :param l: Collapses subword tokenization with ## (could also use the convert token to words functions)
        :return:
        '''
        words = [l[0]]
        for x in range(1, len(l)):
            if '##' in l[x]:
                words[-1] += l[x].replace('##', '')
            else:
                words.append(l[x])
        return words


if __name__ == '__main__':
    # Explaining the influence on sentence length
    IM = InputMarginalizationExplainer(delta=0.05)
    t = BertScore(mode=REFERENCE_FREE)
    print(IM.explain_sentence('Hallo Sie da, was machen Sie denn da, Sie sollten das nicht tun Affe', "Hey you there, what are you doing, you shouldn't do that.", t))
    print(IM.explain_sentence('Hallo du da, was machst du denn da?',
                              "Hey there, my goood friend, what are you doing there?", t))

    print(IM.explain_sentence('Hallo du da, was machst du denn da, ergioijowierjgo?',
                              "Hey there, my goood friend, what are you doing there?", AnyMetric(lambda src, tgt:[len(s+t) for s,t in zip(src, tgt)])))