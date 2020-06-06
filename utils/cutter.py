import os
import json
import thulac

cutter = thulac.thulac(seg_only=True)
frequency = {}

path_list = [
    ["/data/disk3/private/zhx/theme/data/scm/origin",
     "/data/disk3/private/zhx/theme/data/scm/cutted"]
]


def cut(s):
    arr = list(cutter.fast_cut(s))
    for a in range(0, len(arr)):
        arr[a] = arr[a][0]
    for word in arr:
        if not (word in frequency):
            frequency[word] = 0
        frequency[word] += 1
    return arr


if __name__ == "__main__":
    for input_path, output_path in path_list:
        os.makedirs(output_path, exist_ok=True)
        for filename in os.listdir(input_path):
            print(os.path.join(input_path, filename))
            data = []

            f = open(os.path.join(input_path, filename), "r", encoding="utf8")

            for line in f:
                x = json.loads(line)
                for name in ["A", "B", "C"]:
                    x[name] = cut(x[name].replace("\n", ""))

                data.append(x)

            f = open(os.path.join(output_path, filename), "w", encoding="utf8")
            for x in data:
                print(json.dumps(x, ensure_ascii=False, sort_keys=True), file=f)
            f.close()

    json.dump(frequency, open("/data/disk3/private/zhx/theme/data/scm/frequency.txt", "w", encoding="utf8"),
              indent=2,
              ensure_ascii=False)
