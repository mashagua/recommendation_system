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
        tmp=sorted(v,key=lambda _:_[2])
        for i in range(len(tmp)):
            if i<len(tmp)-2:
                train_set.append(str(k)+','+tmp[i][0]+','+tmp[i][1])
            else:
                test_set.append(str(k)+','+tmp[i][0]+','+tmp[i][1])

    return train_set,test_set

train_set,test_set=train_test_split(user_info)
print(train_set[:7])


def save_data(train_set, test_set):
    import random
    random.shuffle(train_set)
    random.shuffle(test_set)
    with open("../data/train_set", "w") as f:
        for line in train_set:
            f.write(line + "\n")

    with open("../data/test_set", "w") as f:
        for line in test_set:
            f.write(line + "\n")

save_data(train_set, test_set)
#定义哈希方法：
def bkdr2hash64(str01):
    mask60=0x0fffffffffffffff
    seed=131
    hash=0
    for s in str01:
        hash=hash*seed+ord(s)
    return hash & mask60

def tohash(file,save_path):
    wfile=open(save_path,'w')
    with open(file) as f:
        for line in f:
            tmp=line.strip().split(',')
            user_id=bkdr2hash64('UserID='+tmp[0])
            movie_id=bkdr2hash64('MovieID='+tmp[1])
            wfile.write(str(user_id)+','+str(movie_id)+','+tmp[2]+'\n')
    wfile.close()

train_file_path='../data/train_set'
train_tohash='../data/train_set_tohash'
test_file_path='../data/test_set'
test_tohash='../data/test_set_tohash'
tohash(train_file_path,train_tohash)
tohash(test_file_path,test_tohash)

import tensorflow as tf
def get_tfrecords_ex(feature,label):
  tfrecords_features={
      "feature":tf.train.Feature(int64_list=tf.train.Int64List(value=feature)),
      "label":tf.train.Feature(float_list=tf.train.FloatList(value=label))
  }
  return tf.train.Example(
      features=tf.train.Features(feature=tfrecords_features)
  )


def totfrecords(file,save_dir):
  print(f"Process to tfrecord file: {file}")
  num=0
  writer=tf.io.TFRecordWriter(save_dir+"/"+"part-0000"+str(num)+".tfrecords")
  lines=open(file)
  for i,line in enumerate(lines):
    tmp=line.strip().split(",")
    feature=[int(tmp[0]),int(tmp[1])]
    label=[float(1) if float(tmp[2])>=3 else float(0)]
    example=get_tfrecords_ex(feature,label)
    writer.write(example.SerializeToString())
    if (i+1)%200000==0:
      writer.close()
      num+=1
      writer=tf.io.TFRecordWriter(save_dir+"/"+"part-0000"+str(num)+".tfrecords")
  print(f"Process to tfrecord file: {file} End")
  writer.close()

import os
train_file_path="../data/train_set_tohash"
train_totfrecord=os.path.join("../data","train")
test_file_path=os.path.join("../data","test_set_tohash")
test_totfrecord=os.path.join("../data","val")
# os.mkdir(train_totfrecord)
# os.mkdir(test_totfrecord)
totfrecords(train_file_path,train_totfrecord)
totfrecords(test_file_path,test_totfrecord)






# if __name__=='__main__':
