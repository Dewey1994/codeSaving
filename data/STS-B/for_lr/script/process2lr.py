import pandas as pd
import os

sts_path = r'/nlp/data/STS-B/raw'
dirs = os.listdir(sts_path)
for path in dirs:
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), sts_path, path), 'r', encoding='utf-8') as f:

        x = f.readlines()
        text = []
        label = []
        rests = []
        scores = []
        for i in x:
            res = i.strip().split('||')

            text.append(res[1][:-1])  # 去掉句号
            label.append(res[2][:-1])
            scores.append(res[3])

        df = pd.DataFrame({"data": text, "label": label, "score": scores})
        df.to_csv(path.replace('txt', 'csv'), index=False)



