import multiprocessing
from pyspark.sql import SparkSession
import pyspark
from legacy.vocab import load_vocab
from pathlib import Path
import json
import numpy as np
import tensorflow as tf
# import tensorflow_cpu as tf
from random import random, randint


# def split_list(l, val):
#     res = []
#     while val in l:
#         i = l.index(val)
#         res.append(l[:i])
#         l = l[i + 1:]
#     res.append(l)
#     return res
#
#
# def all_indices(l, val):
#     res = []
#     count = 0
#     while val in l:
#         i = l.index(val)
#
#         res.append(count + i)
#         count += len(l[:i]) + 1
#         l = l[i + 1:]
#     return res


# def mlm_preprocess(labels, temp_t, vocab):
#     labels.append(tf.convert_to_tensor(temp_t, dtype=tf.int32))
#     for i in range(len(temp_t)):
#         if random() < 0.15:
#             chance = random()
#             if chance < 0.8:
#                 # 80% masked token
#                 temp_t[i] = vocab["[MASK]"]
#             elif chance < 0.9:
#                 temp_t[i] = randint(5, len(vocab) - 1)
#     return labels, temp_t
#
#
# def sop_preprocess(temp_s, temp_t, labels, delim):
#     # closest ";" separator to center
#     fragments = all_indices(temp_t, delim)
#     # if there is single sentence
#     if len(fragments) != 0:
#         closest_index = min([(abs(x - len(temp_t) // 2), x) for x in fragments])[1]
#         left = temp_t[:closest_index]
#         right = temp_t[closest_index + 1:]
#         if random() > 0.5: # no swap
#             labels.append(tf.convert_to_tensor([0.0,1.0], dtype=tf.float64))
#             temp_s = [0 for _ in range(len(left))] + [1 for _ in range(len(right) + 1)]
#             temp_t = left + [delim] + right
#         else: # swap
#             labels.append(tf.convert_to_tensor([1.0,0.0], dtype=tf.float64))
#             temp_s = [0 for _ in range(len(right))] + [1 for _ in range(len(left) + 1)]
#             temp_t = right + [delim] + left
#     else:
#         temp_s = []
#         temp_t = []
#     return temp_s, temp_t, labels


def make_spark_session():
    cpu_count = multiprocessing.cpu_count()

    spark = SparkSession.builder\
        .master("local[*]")\
        .appName("SparkApp") \
        .config("spark.driver.memory","7g") \
        .config("spark.executor.memory", "7g") \
        .config("spark.driver.maxResultSize", "3g") \
        .config("spark.memory.offHeap.enabled", "true") \
        .config("spark.memory.offHeap.size", "1g") \
        .config("spark.cleaner.periodicGC.interval", "1min") \
        .getOrCreate()
    sc = spark.sparkContext

    conf = pyspark.SparkConf().setAll([
        ('spark.executor.cores', str(cpu_count)),
        ('spark.cores.max', str(cpu_count))])
    spark.sparkContext.stop()
    # spark = SparkSession.builder.config(conf=conf).getOrCreate()
    # sc = spark.sparkContext
    # print(sc.getConf().getAll())
    return conf#spark, sc
import os

# def tokenize_file(data):
#     vocab, paths, max_len, mlm, sop = data
#     tokens = []
#     segments = []
#
#     tokenizer = MusicTokenizer(vocab)
#
#     for path in paths:
#         size = os.path.getsize(path)
#         max_Mb = 0.4
#         Mb = 2**20
#         if size <= Mb * max_Mb:
#             # try:
#             token, segment = tokenizer.tokenize_file(path, max_len, mlm, sop)
#
#             token = tf.stack(token, axis=0)
#             segment = tf.stack(segment, axis=0)
#             #     if len(token) == max_len:
#             tokens.append(token)
#             segments.append(segment)
#
#             del token, segment
#             # except Exception as e:
#             #     print(e,'\n',path,"SEEMS TO BE TOO BIG", size)
#         else:
#             print(path,"(",size, ") is too large", size <= Mb * max_Mb, Mb * max_Mb)
#     del tokenizer
#     del vocab, paths, max_len, mlm, sop
#     del data
#     return (tokens, segments)


import gc
from pyspark.sql import SQLContext
# from multiprocessing_file_defs import tokenize_file
from legacy.multiprocessing_part_defs import tokenize_part
from multiprocessing import Pool
# from billiard.pool import Pool


def make_dataset(res_path,
                 vocab_file = r"./vocab.json",
                 data_dirs = [],
                 batch_size = 1000,
                 max_len=64,
                 sop=False,
                 mlm=False,
                 nparts = 5):
    # tokenizer = MusicTokenizer(load_vocab(vocab_file))
    # tokens, labels
    # dataset = \
    # tokenizer.tokenize_folders(data_dirs,res_path,
    tokenize_folders(load_vocab(vocab_file),
                     data_dirs,
                     res_path,
                    nparts=nparts,
                    batch_size=batch_size,
                    sop=sop,
                    max_len=max_len,
                    mlm=mlm)
    # if sop or mlm:
    #     dataset = tf.data.Dataset.from_tensor_slices((tokens, labels))
    # else:
    #     dataset = tf.data.Dataset.from_tensor_slices(tokens)
    # return dataset


def save_labeled_dataset(dataset, folder):
    token_dataset = dataset.map(lambda token, label: token)
    label_dataset = dataset.map(lambda token, label: label)

    token_dataset = list(token_dataset.as_numpy_iterator())
    label_dataset = list(label_dataset.as_numpy_iterator())
    #     print(token_dataset)

    token_dataset = tf.stack(token_dataset, axis=0).numpy()
    #     print(tf.comvert_to_tensor())
    #     print(token_dataset)

    label_dataset = tf.stack(label_dataset, axis=0).numpy()

    np.save(folder + r"/tokens.npy", token_dataset)
    np.save(folder + r"/labels.npy", label_dataset)


def load_labeled_dataset(folder):
    token_dataset = np.load(folder + r"/tokens.npy")
    label_dataset = np.load(folder + r"/labels.npy")
    # print(label_dataset)
    token_dataset = tf.convert_to_tensor(token_dataset)
    label_dataset = tf.convert_to_tensor(label_dataset)

    dataset_restored = tf.data.Dataset.from_tensor_slices((token_dataset, label_dataset))
    return dataset_restored

import multiprocessing
# We must import this explicitly, it is not imported by the top-level
# multiprocessing module.
import multiprocessing.pool
import time

from random import randint


class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class MyPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess

def tokenize_folders(vocab, folders, res_path, batch_size = 1000, max_len=512,
                     mlm=False, sop=False, nparts = 5):
    # tf.get_logger().setLevel('ERROR')

    count = 0
    paths = []
    for folder in folders:
        paths += [str(x) for x in Path(folder).glob("*.txt")]

    part_size = len(paths) // nparts
    r = range(len(paths))
    parts = list([paths[r:r + part_size] for r in r[::part_size]])
    print(len(parts), "path groups", part_size, "files each")

    count += 1
    files = [(vocab, parts[i], max_len, mlm, sop, batch_size, i + 1, res_path) for i in range(len(parts))]
    # print(files[0])
    cpu_count = multiprocessing.cpu_count()
    with MyPool(2) as p:
        p.map(tokenize_part, files)

    # for paths in parts:
        # tokens = []
        # labels = []
        #
        # count += 1
        # r = range(len(paths))
        # paths = list([paths[r:r + batch_size] for r in r[::batch_size]])
        #
        # files = [(vocab, path, max_len, mlm, sop) for path in paths]
        # del paths
        #
        # cpu_count = multiprocessing.cpu_count()
        #
        # print("Group", count, ": consists of",batch_size ,"in each batch, on", cpu_count, "CPUs")
        #
        # with Pool(cpu_count) as p:
        #     res = p.map(tokenize_file, files)
        # del files
        # print("res type",type(res))
        #
        # for i in range(len(res)):
        #     token, label = res[i]
        #     for i in range(len(token)):
        #         if token[i].dtype == tf.int32:
        #             tokens.append(token[i])
        #             labels.append(label[i])
        #
        # tokens = tf.concat(tokens, axis=0)
        # labels = tf.concat(labels, axis=0)
        #
        # # return tokens, labels
        # if sop or mlm:
        #     dataset = tf.data.Dataset.from_tensor_slices((tokens, labels))
        # else:
        #     dataset = tf.data.Dataset.from_tensor_slices(tokens)
        #
        # del tokens
        # del labels
        # dataset_path = res_path + r"\part_" + str(count)
        # try:
        #     os.mkdir(dataset_path)
        # except:
        #     print("Already exists:",dataset_path)
        # save_labeled_dataset(dataset, dataset_path)
        # del dataset

