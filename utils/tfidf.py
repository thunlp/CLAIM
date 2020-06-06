import json
import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

train_path = "/data/disk3/private/zhx/theme/data/scm/origin/train.json"
valid_path = "/data/disk3/private/zhx/theme/data/scm/origin/valid.json"


def trans(x):
    x = list(jieba.cut(x))
    return " ".join(x)


if __name__ == "__main__":
    se = set()
    f = open(train_path, "r", encoding="utf8")
    for line in f:
        x = json.loads(line)
        se.add(x["A"])
        se.add(x["B"])
        se.add(x["C"])

    f = open(valid_path, "r", encoding="utf8")
    for line in f:
        x = json.loads(line)
        se.add(x["A"])
        se.add(x["B"])
        se.add(x["C"])

    data = list(se)
    for a in range(0, len(data)):
        data[a] = trans(data[a])

    tfidf_model = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b").fit(data)
    sparse_result = tfidf_model.transform(data)

    f = open(valid_path, "r", encoding="utf8")
    cnt = 0
    total = 0
    for line in f:
        x = json.loads(line)
        y = [
            trans(x["A"]),
            trans(x["B"]),
            trans(x["C"])
        ]

        y = tfidf_model.transform(y)
        y = y.todense()

        v1 = np.sum(np.dot(y[0], np.transpose(y[1])))
        v2 = np.sum(np.dot(y[0], np.transpose(y[2])))
        if v1 > v2:
            ans = "B"
        else:
            ans = "C"

        if ans == x["label"]:
            cnt += 1
        total += 1

    print(cnt, total, cnt / total)
