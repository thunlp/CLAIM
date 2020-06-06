import json
import os
import random

input_path = "/data/disk3/private/zhx/theme/data/element/origin"
output_path = "/data/disk3/private/zhx/theme/data/element/format"

if __name__ == "__main__":
    for dir_name in os.listdir(input_path):
        filename = "data.json"
        f = open(os.path.join(input_path, dir_name, filename), "r", encoding="utf8")
        data = []
        for line in f:
            data = data + json.loads(line)
        random.shuffle(data)

        os.makedirs(os.path.join(output_path, dir_name), exist_ok=True)
        f_train = open(os.path.join(output_path, dir_name, "train.json"), "w", encoding="utf8")
        f_test = open(os.path.join(output_path, dir_name, "test.json"), "w", encoding="utf8")
        for x in data:
            if random.randint(1, 5) == 1:
                print(json.dumps(x, ensure_ascii=False, sort_keys=True), file=f_test)
            else:
                print(json.dumps(x, ensure_ascii=False, sort_keys=True), file=f_train)

        f_test.close()
        f_train.close()
