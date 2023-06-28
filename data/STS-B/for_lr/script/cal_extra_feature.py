import jieba
import pandas as pd


def get_len_diff(sten1, sten2):
    return 1 - abs(len(sten1) - len(sten2)) / float(max(len(sten1), len(sten2)))


def get_edit_distance(sten1, sten2):
    m = len(sten1) + 1
    n = len(sten2) + 1
    f = [[0] * n for j in range(m)]
    # f1 = [[0 for i in range(n)] for j in range(m)]
    for i in range(1, m):
        f[i][0] = i
    for j in range(1, n):
        f[0][j] = j
    for i in range(1, m):
        for j in range(1, n):
            if sten1[i-1] == sten2[j-1]:
                f[i][j] = f[i - 1][j - 1]
            else:
                f[i][j] = min(f[i - 1][j], f[i][j - 1], f[i - 1][j - 1]) + 1
    return 1 - f[m - 1][n - 1] / float(max(len(sten1), len(sten2)))


def get_word_diff(sten1, sten2):
    words1 = set(jieba.cut(sten1))
    words2 = set(jieba.cut(sten2))
    same_word_len = len(words1 & words2)
    unique_words1 = len(words1 - words2)
    unique_words2 = len(words2 - words1)
    avg_len = (len(words1) + len(words2)) / 2
    max_len = max(len(words1), len(words2))
    min_len = min(len(words1), len(words2))
    jaccard_sim = same_word_len / float(len(words1 | words2))

    return same_word_len / max_len, same_word_len / min_len, same_word_len / avg_len, \
           unique_words1 / len(words1), unique_words2 / len(words2), jaccard_sim


if __name__ == '__main__':
    train_data_path = r"/data/STS-B/for_lr/cnsd-sts-train.csv"
    dev_data_path = r"/data/STS-B/for_lr/cnsd-sts-dev.csv"
    test_data_path = r"/data/STS-B/for_lr/cnsd-sts-test.csv"
    paths = [train_data_path, dev_data_path, test_data_path]

    for i, p in enumerate(paths):
        text1 = []
        text2 = []
        val = []
        len_diff = []
        edit_distance = []
        same1 = []
        same2 = []
        same3 = []
        uni1 = []
        uni2 = []
        jaccard = []
        df = pd.read_csv(p, sep=',')
        rows = df.to_dict('reocrds')
        for idx, row in enumerate(rows):
            data = row['data'].strip()
            label = row['label'].strip()
            score = float(row['score'])
            text1.append(data)
            text2.append(label)
            val.append(score)
            len_diff.append(get_len_diff(data, label))
            edit_distance.append(get_edit_distance(data, label))
            same1.append(get_word_diff(data, label)[0])
            same2.append(get_word_diff(data, label)[1])
            same3.append(get_word_diff(data, label)[2])
            uni1.append(get_word_diff(data, label)[3])
            uni2.append(get_word_diff(data, label)[4])
            jaccard.append(get_word_diff(data, label)[5])
        if i == 0:
            embed = []
            with open('train_embed.txt', 'r', encoding='utf-8') as f:
                x = f.readlines()
                for i in x:
                    embed.append(float(i.strip()))

            dfs = pd.DataFrame({"data":text1,"label":text2,"len_diff":len_diff,"edit":edit_distance,
                                "same1":same1,"same2":same2,"same3":same3,"uni1":uni1,"uni2":uni2,"jaccard":jaccard,
                                "embed":embed,"score":val})
            dfs.to_csv("train-lr.csv",index=False)
        elif i == 1:
            embed = []
            with open('dev_embed.txt', 'r', encoding='utf-8') as f:
                x = f.readlines()
                for i in x:
                    embed.append(float(i.strip()))
            dfs = pd.DataFrame(
                {"data": text1, "label": text2, "len_diff": len_diff, "edit": edit_distance,
                 "same1": same1, "same2": same2, "same3": same3, "uni1": uni1, "uni2": uni2, "jaccard": jaccard,
                 "embed":embed,"score":val})
            dfs.to_csv("dev-lr.csv",index=False)
        else:
            embed = []
            with open('test_embed.txt', 'r', encoding='utf-8') as f:
                x = f.readlines()
                for i in x:
                    embed.append(float(i.strip()))
            dfs = pd.DataFrame({"data": text1, "label": text2, "len_diff": len_diff, "edit": edit_distance,
                                "same1": same1, "same2": same2, "same3": same3, "uni1": uni1, "uni2": uni2,
                                "jaccard": jaccard,
                                "embed": embed, "score": val})
            dfs.to_csv("test-lr.csv",index=False)
