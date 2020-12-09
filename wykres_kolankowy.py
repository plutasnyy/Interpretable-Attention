from ast import literal_eval

import numpy as np
import pandas as pd
import plotly.express as px
from tqdm import tqdm

from calculate_threshold import preds_to_spans, f1
from predict_last_model import OrthoModel

if __name__ == '__main__':
    model = OrthoModel()
    trial = pd.read_csv("../../data/spans/tsd_train.csv")
    trial["spans"] = trial.spans.apply(literal_eval)
    preds = list()

    for i, row in tqdm(trial.iterrows(), total=len(trial)):
        if len(row['text']) < 663:
            prediction = model.predict(row['text'])
            preds.append([*prediction, row['spans']])

    data = list()
    for threshold in np.linspace(0, 1, 100):
        f1_score = []
        for pred in preds:
            *prediction, grand_truth_spans = pred
            predicted_spans = preds_to_spans(prediction, threshold=threshold, cumulative=False)
            score = f1(predicted_spans, grand_truth_spans)
            f1_score.append(score)
        data.append([threshold, np.mean(np.array(f1_score))])
    df = pd.DataFrame(data, columns=['threshold', 'f1'])
    print(df.head(10))
    fig = px.scatter(df, x="threshold", y="f1")
    fig.write_image('kolanko_not_cumulative_less663.png')
