import argparse
import os

import numpy as np
from nltk import TreebankWordTokenizer

from Trainers.DatasetBC import datasets
from common_code.common import get_latest_model
from configurations import configurations
from model.Binary_Classification import Model
from preprocess.vectorizer import UNK, EOS, SOS


class OrthoModel():
    def __init__(self):
        args = argparse.Namespace(attention='tanh', data_dir='.', dataset='civil', diversity=0, encoder='ortho_lstm',
                                  extras=[], output_dir='./experiments')

        self.dataset = datasets[args.dataset](args)

        if args.output_dir is not None:
            self.dataset.output_dir = args.output_dir

        self.dataset.diversity = args.diversity
        config = configurations[args.encoder](self.dataset)
        latest_model = get_latest_model(os.path.join(config['training']['basepath'], config['training']['exp_dirname']))
        self.model = Model.init_from_config(latest_model, load_gen=False)

    def predict(self, text):
        spans = list(TreebankWordTokenizer().span_tokenize(text))
        list_of_tokens = [text[i:j] if text[i:j] in self.dataset.vec.word2idx else UNK for (i, j) in spans]
        tokenized_spans = [None, *spans, None]
        list_of_tokens = [SOS, *list_of_tokens, EOS]
        sequences = list(map(lambda s: int(self.dataset.vec.word2idx[s]), list_of_tokens))
        predictions, attentions, conicity_values = self.model.evaluate([sequences])
        predictions = np.array(predictions)
        return predictions[0], attentions[0], tokenized_spans, list_of_tokens

# idx = 0
# print('prediction', predictions[idx])
# result = ''
# last_span = 0
# for word, attn, span in zip(list_of_tokens, attentions[idx], tokenized_spans):
#     if span is not None:
#         i, j = span
#         if i > last_span:
#             result += ' '
#         last_span = j
#         print(word, round(attn, 4), (i, j))
#         if attn > 0.2:
#             result += colored(text[i:j], 'red', attrs=['reverse', 'blink'])
#         else:
#             result += text[i:j]
#
# print(result)
