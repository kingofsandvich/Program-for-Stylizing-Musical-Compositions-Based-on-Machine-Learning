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
    return conf

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
    tokenize_folders(load_vocab(vocab_file),
                     data_dirs,
                     res_path,
                    nparts=nparts,
                    batch_size=batch_size,
                    sop=sop,
                    max_len=max_len,
                    mlm=mlm)


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

    cpu_count = multiprocessing.cpu_count()
    with MyPool(2) as p:
        p.map(tokenize_part, files)


