import pandas as pd
import os
import jieba

from nlp.text_similarity.bm25 import utils, bm25

sts_path = r'/data/STS-B/raw'
dirs = os.listdir(sts_path)
for path in dirs:
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), sts_path, path), 'r', encoding='utf-8') as f:
        if 'train' in path:
            x = f.readlines()
            text = []
            label = []
            rests = []
            hard_neg = []
            for i in x:
                res = i.strip().split('||')
                if res[3]=='5' or res[3]=='4':
                    text.append(res[1][:-1]) # 去掉句号
                    label.append(res[2][:-1])
                else:
                    rests.append(res[1])
                    rests.append(res[2])
            doc = []
            for sent in rests:
                words = list(jieba.cut(sent))
                words = utils.filter_stop(words)
                doc.append(words)
            # print(doc)
            s = bm25.BM25(doc)
            # print(s.f)
            # print(s.idf)
            for t in text:
                res = s.simall(utils.filter_stop(jieba.cut(t)))
                sort_res = sorted(res,key=lambda x:x[0],reverse=True)
                hard_neg.append(sort_res[1][1])
            df = pd.DataFrame({"data":text,"label":label,"hard_neg":hard_neg})
            df.to_csv(path.replace('txt','csv'),index=False)
        else:
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



