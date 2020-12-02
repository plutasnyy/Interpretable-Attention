import argparse
import os

from nltk import word_tokenize, TreebankWordTokenizer
from nltk.tokenize.api import StringTokenizer
from termcolor import colored

from Trainers.DatasetBC import datasets
from Trainers.TrainerBC import Evaluator
from common_code.common import get_latest_model
from configurations import configurations
from model.Binary_Classification import Model
import numpy as np

from preprocess.vectorizer import cleaner, UNK, EOS, SOS

parser = argparse.ArgumentParser(description='Run experiments on a dataset')
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument("--output_dir", type=str)
parser.add_argument('--encoder', type=str, choices=['vanilla_lstm', 'ortho_lstm', 'diversity_lstm'], required=True)
parser.add_argument("--diversity", type=float, default=0)

args, extras = parser.parse_known_args()
args.extras = extras
args.attention = 'tanh'

dataset = datasets[args.dataset](args)

if args.output_dir is not None:
    dataset.output_dir = args.output_dir

dataset.diversity = args.diversity
config = configurations[args.encoder](dataset)
latest_model = get_latest_model(os.path.join(config['training']['basepath'], config['training']['exp_dirname']))
model = Model.init_from_config(latest_model, load_gen=False)

text = 'you ignorant sheep need serious help for your stupidity \nhttps://www.youtube.com/watch?v=SXxHfb66ZgM'.lower()
print(text)
spans = list(TreebankWordTokenizer().span_tokenize(text))
list_of_tokens = [text[i:j] if text[i:j] in dataset.vec.word2idx else UNK for (i, j) in spans]
tokenized_spans = [None, *spans, None]
list_of_tokens = [SOS, *list_of_tokens, EOS]
sequences = list(map(lambda s: int(dataset.vec.word2idx[s]), list_of_tokens))
predictions, attentions, conicity_values = model.evaluate([sequences])
predictions = np.array(predictions)

idx = 0
print('prediction', predictions[idx])
result = ''
last_span = 0
for word, attn, span in zip(list_of_tokens, attentions[idx], tokenized_spans):
    if span is not None:
        i, j = span
        if i > last_span:
            result += ' '
        last_span = j
        print(word, round(attn, 4), (i, j))
        if attn > 0.2:
            result += colored(text[i:j], 'red', attrs=['reverse', 'blink'])
        else:
            result += text[i:j]

print(result)
