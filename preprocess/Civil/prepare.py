import re

import pandas as pd
from tqdm import tqdm

from preprocess.vectorizer import cleaner

data = []


def cleaner_20(text):
    text = cleaner(text)
    text = re.sub(r'(\W)+', r' \1 ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


for stage in tqdm(['test', 'train']):
    df = pd.read_csv('../../../../data/civil_comments/' + stage + '.csv')
    for index, row in df.iterrows():
        text = cleaner_20(str(row['comment_text']))
        text = text.replace('\n', '')
        y = int(row['toxicity'] > 0.5)
        data.append([text, y, stage])


result_df = pd.DataFrame(data, columns=['text', 'label', 'exp_split'])
result_df.to_csv('civil_dataset.csv', index=False)
