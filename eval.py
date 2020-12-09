from ast import literal_eval

import numpy as np
import pandas as pd
from tqdm import tqdm

from calculate_threshold import preds_to_spans, f1
from predict_last_model import OrthoModel

trial = pd.read_csv("../../data/spans/tsd_trial.csv")
trial["spans"] = trial.spans.apply(literal_eval)
data = list()

model = OrthoModel()
for i, row in tqdm(trial.iterrows(), total=len(trial)):
    preds = model.predict(row['text'])
    predicted_spans = preds_to_spans(preds, threshold=0.5, cumulative=True)
    score = f1(predicted_spans, row['spans'])
    data.append(score)

print(np.mean(np.array(data)))
