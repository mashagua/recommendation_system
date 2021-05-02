import numpy as np

with open('../data/ratings.dat') as f:
    lines = f.readlines()
print(len(lines))


def read_raw_data(file_path):
    user_info = {}
    lines = open(file_path)
    for line in lines:
        tmp = line.strip().split("::")
        # remove irregular data
        if len(tmp) < 4:
            continue
        ui = user_info.get(tmp[0], None)
        # first occur
        if ui is None:
            user_info[tmp[0]] = [(tmp[1], tmp[2], tmp[3])]
        else:
            user_info[tmp[0]].append((tmp[1], tmp[2], tmp[3]))
    return user_info


user_action_num = {}
user_info = read_raw_data('../data/ratings.dat')
for k, v in user_info.items():
    user_action_num[k] = len(v)
user_stat = np.asarray(list(user_action_num.values()))
max_num = np.max(user_stat)
min_num = np.min(user_stat)
median_num = np.median(user_stat)
average_num = np.average(user_stat)
print(max_num, min_num, median_num, average_num)
# q1:抽取用户小于2000


def extract_valid_user(user_info):
    user_info_filter = {}
    for k, v in user_info.items():
        if len(v) > 2000:
            continue
        user_info_filter[k] = v
    return user_info_filter


print(f"总的用户量：{len(user_info)}")


def train_test_split(user_info):
    train_set = []
    test_set = []
    for k,v in user_info.items():
        tmp=sorted(v,k=lambda _:_[2])
        for i in range(len(tmp)):
            if i<len(tmp)-2:
                train_set.append(str(k)+','+tmp[i][0]+','+tmp[i][1])
            else:
                test_set.append(str(k)+','+tmp[i][0]+','+tmp[i][1])

    return train_set,test_set




# if __name__=='__main__':
