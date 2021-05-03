import tensorflow as tf
import numpy as np
import os


class Singleton(type):
    _instance = {}

    def __call__(cls, *args, **kwargs):
        if cls not in Singleton._instance:
            Singleton._instance[cls] = type.__call__(cls, *args, **kwargs)
        return Singleton._instance[cls]


class PS(metaclass=Singleton):
    def __init__(self, embedding_dim):
        np.random.seed(2021)
        self.dim = embedding_dim
        self.parameter_sever = {}
        print("ps inited...")

    # 拉取特征对应的参数,len+batch,二维矩阵，从服务器中取出
    def pull(self, keys):
        values = []
        for k in keys:
            tmp = []
            # 传进来的是【batch,feature_len】,是个二维矩阵，对二维矩阵进行循环遍历
            for arr in k:
                value = self.parameter_sever.get(arr, None)
                if value is None:
                    value = np.random.rand(self.dim)
                    self.parameter_sever[arr] = value
                tmp.append(value)
            values.append(tmp)
        return np.asarray(values, dtype="float32")

# 回写到参数服务器中
    def push(self, keys, values):
        # 回写入参数服务器中
        for i in range(len(keys)):
            for j in range(len(keys[i])):
                self.parameter_sever[keys[i][j]] = values[i][j]

    def delete(self, keys):
        for k in keys:
            self.parameter_sever.pop(k)

    def save(self, path):
        print(f"总共包含keys {len(self.parameter_sever)}")
        writer = open(path, "w")
        for k, v in self.parameter_sever.items():
            writer.write(str(k) + "\t" +
                         ",".join(["%.8f" % _ for _ in v]) + "\n")
        writer.close()


tf.compat.v1.disable_eager_execution()


class InputFn:
    def __init__(self, local_ps):
        self.feature_len = 2
        self.label_len = 1
        self.n_parse_threads = 4
        self.shuffle_buffer_size = 1024
        self.prefetch_buffer_size = 1
        self.batch = 8
        self.local_ps = local_ps

    def input_fn(self, data_dir, is_test=False):
        def __parse_example(example):
            features = {
                "feature": tf.io.FixedLenFeature(self.feature_len, tf.int64),
                "label": tf.io.FixedLenFeature(self.label_len, tf.float32)
            }
            return tf.io.parse_single_example(example, features)

        def __get_embedding(parsed):
            keys = parsed['feature']
            keys_array = tf.compat.v1.py_func(
                self.local_ps.pull, [keys], tf.float32)
            results = {
                'feature': parsed['feature'],
                'label': parsed['label'],
                'feature_embedding': keys_array
            }
            return results
        file_list = os.listdir(data_dir)
        files = []
        for i in range(len(file_list)):
            files.append(os.path.join(data_dir, file_list[i]))
        dataset = tf.compat.v1.data.Dataset.list_files(files)
        if is_test:
            dataset = dataset.repeat(1)
        else:
            dataset = dataset.repeat()
        dataset = dataset.interleave(
            lambda _: tf.compat.v1.data.TFRecordDataset(_),
            cycle_length=1)


if __name__ == "__main__":
    # 引向量的维度是8
    ps_local = PS(8)
    keys = [[123, 234], [567, 891]]
    res = ps_local.pull(keys)
    print(f"参数服务器中有哪些参数 \n {ps_local.parameter_sever}")
    print(f"keys获取到的对应向量 {res}")
    gradient = 10
    res = res - 0.01 * gradient
    ps_local.push(keys, res)
    path = os.path.join("../data", "feature_embedding")
    ps_local.save(path)
