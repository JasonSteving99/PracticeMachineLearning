import numpy as np 


ZOO_FEATURE_DIMENSIONS = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 6, 2, 2, 2, 7]
ZOO_LEGS_FEATURE_INDEX = 12
ZOO_LEGS_FEATURE_MAPPING = lambda x: {0:0,2:1,4:2,5:3,6:4,8:5}[x]
ZOO_TYPE_FEATURE_INDEX = 16
ZOO_TYPE_FEATURE_MAPPING = lambda x: x - 1


def parseZooDataLine(zoo_data_line):
    zoo_data_ints = np.array(list(map(int, list(filter(lambda c: c != '', zoo_data_line.strip()[1:-1].split(' ')))[1:])))
    zoo_data_ints[ZOO_LEGS_FEATURE_INDEX] = ZOO_LEGS_FEATURE_MAPPING(zoo_data_ints[ZOO_LEGS_FEATURE_INDEX])
    zoo_data_ints[ZOO_TYPE_FEATURE_INDEX] = ZOO_TYPE_FEATURE_MAPPING(zoo_data_ints[ZOO_TYPE_FEATURE_INDEX])
    return zoo_data_ints


def parseZooData():
    with open('zoo.data.txt', 'r') as f:
        zoo_data = np.array(list(map(parseZooDataLine, filter(lambda line: len(line.strip()) > 0, f.readlines()))))
        return zoo_data


if __name__ == '__main__':
    for zoo_data_sample in parseZooData():
        print(zoo_data_sample)
   
