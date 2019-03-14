import math


def NME(x1, y1):
    Accumulate = 0
    for i in range(106):
        temp1 = x1[i] - y1[i]
        temp2 = x1[i + 1] - y1[i + 1]
        rate = math.sqrt(temp1 ** 2 + temp2 ** 2)
        rate = rate / 450.0
        Accumulate = Accumulate + rate

    Accumulate = Accumulate / 106.0
    return Accumulate


if __name__ == '__main__':
    result = []
    gt = []
    with open('val.txt') as f:
        for line in f.readlines():
            rows = list(map(float, line.strip().split(' ')[1:]))
            gt.append(rows)

    with open('result50_face_final.txt') as f:
        for line in f.readlines():
            rows = list(map(float, line.strip().split(' ')[1:]))
            result.append(rows)

    error = 0
    for i in range(len(gt)):
        error = NME(result[i], gt[i]) + error
    print(error)
