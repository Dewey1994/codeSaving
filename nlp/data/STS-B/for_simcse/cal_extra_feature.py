import jieba


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
            if sten1[i] == sten2[j]:
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
