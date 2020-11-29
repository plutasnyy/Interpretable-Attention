import argparse
import os

from Trainers.DatasetBC import datasets
from Trainers.TrainerBC import Evaluator
from common_code.common import get_latest_model
from configurations import configurations

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
evaluator = Evaluator(dataset, latest_model, _type=dataset.trainer_type)
y_pred, attentions = evaluator.evaluate(dataset.test_data, save_results=True)


idx = 0
sentence = dataset.vec.map2words(dataset.test_data.X[idx])
for word, attn in zip(sentence, attentions[idx]):
    print(word, round(attn, 4))

print('aaa')