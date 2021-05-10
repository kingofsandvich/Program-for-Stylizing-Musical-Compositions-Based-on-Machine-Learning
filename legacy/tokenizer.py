import os
from pathlib import Path
from tokenizers.pre_tokenizers  import Whitespace

# from tokenizers import ByteLevelBPETokenizer
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

import tensorflow as tf
import numpy as np
from transformers import AlbertConfig, TFAlbertModel,TFAlbertForMaskedLM
from random import random, randint

def make_tokenizer(texts_dir="D:\\ВКР\\dataset\\big midi\\textLakh",
                   tokenizer_dir="./musicBPE/"):
    paths = [str(x) for x in Path(texts_dir).glob("*.txt")]

    # Initialize a tokenizer
    tokenizer = ByteLevelBPETokenizer()
    # tokenizer.pre_tokenizer =

    # Customize training
    tokenizer.train(files=paths, vocab_size=52_000, min_frequency=1, special_tokens=[
        "<cls>",
        "<pad>",
        "<sep>",
        "<unk>",
        "<mask>",
    ])

    # Save files to disk
    tokenizer.save(tokenizer_dir)


def get_tokenizer(tokenizer_dir=r'C:\Users\1\Desktop\musicBPE'):
    tokenizer = ByteLevelBPETokenizer(
        vocab_file=tokenizer_dir + r"\vocab.json",
        merges_file=tokenizer_dir + r"\merges.txt"
    )

    tokenizer.enable_truncation(max_length=512)
    return tokenizer


def split_list(l, val):
    res = []
    while val in l:
        i = l.index(val)
        res.append(l[:i])
        l = l[i + 1:]
    res.append(l)
    return res


def all_indices(l, val):
    res = []
    count = 0
    while val in l:
        i = l.index(val)

        res.append(count + i)
        count += len(l[:i]) + 1
        l = l[i + 1:]
    return res


def mlm_preprocess(labels, temp_t, vocab):
    labels.append(tf.convert_to_tensor(temp_t, dtype=tf.int32))
    for i in range(len(temp_t)):
        if random() < 0.15:
            chance = random()
            if chance < 0.8:
                # 80% masked token
                temp_t[i] = vocab["[MASK]"]
            elif chance < 0.9:
                temp_t[i] = randint(5, len(vocab) - 1)
    return labels, temp_t


def sop_preprocess(temp_s, temp_t, labels, delim):
    # closest ";" separator to center
    fragments = all_indices(temp_t, delim)
    # if there is single sentence
    if len(fragments) != 0:
        closest_index = min([(abs(x - len(temp_t) // 2), x) for x in fragments])[1]
        left = temp_t[:closest_index]
        right = temp_t[closest_index + 1:]
        if random() > 0.5: # no swap
            labels.append(tf.convert_to_tensor([0.0,1.0], dtype=tf.float64))
            temp_s = [0 for _ in range(len(left))] + [1 for _ in range(len(right) + 1)]
            temp_t = left + [delim] + right
        else: # swap
            labels.append(tf.convert_to_tensor([1.0,0.0], dtype=tf.float64))
            temp_s = [0 for _ in range(len(right))] + [1 for _ in range(len(left) + 1)]
            temp_t = right + [delim] + left
    else:
        temp_s = []
        temp_t = []
    return temp_s, temp_t, labels



class MusicTokenizer(object):

    def __init__(self, vocab={}):
        self.vocab = vocab
        self.inv_vocab = dict((vocab[k], k) for k in vocab.keys())
        self.cls = "[CLS] "
        self.pad = "[PAD]"
        self.sep = " [SEP] "
        self.unk = "[UNK]"
        self.mask = "[MASK]"

    def join(self, texts=[]):
        return self.cls + self.sep.join(texts)

    # def tokenize_folders(self, folders, res_path, batch_size = 1000, max_len=512,
    #                      mlm=False, sop=False, nparts = 5):
    #     count = 0
    #     paths = []
    #     for folder in folders:
    #         paths += [str(x) for x in Path(folder).glob("*.txt")]
    #
    #
    #     part_size = len(paths) // nparts
    #     r = range(len(paths))
    #     parts = list([paths[r:r + part_size] for r in r[::part_size]])
    #     print(len(parts), "path groups", part_size, "files each")
    #
    #     # spark, sc = make_spark_session()
    #     # conf = make_spark_session()
    #     # spark = SparkSession.builder.config(conf=conf).getOrCreate()
    #     # sc = spark.sparkContext
    #     # print(sc.getConf().getAll())
    #     # sqlContext = SQLContext(sc)
    #     for paths in parts:
    #         tokens = []
    #         labels = []
    #
    #         count += 1
    #         r = range(len(paths))
    #         paths = list([paths[r:r + batch_size] for r in r[::batch_size]])
    #
    #
    #         files = [(self.vocab, path, max_len, mlm, sop) for path in paths]
    #         cpu_count = multiprocessing.cpu_count()
    #
    #         print("Group", count, ": consists of",batch_size ,"in each batch, on", cpu_count, "CPUs")
    #         # rdd = spark.sparkContext.parallelize(files, cpu_count)
    #         # rdd.unpersist(blocking=True)
    #
    #         # rdd_map = rdd.map(tokenize_file)
    #         with Pool(cpu_count) as p:
    #             res = p.map(tokenize_file, files)
    #
    #         # rdd.unpersist(blocking=True)
    #
    #         # print("rdd type",type(rdd))
    #         # print("rdd_map type",type(rdd_map))
    #         # res = rdd_map.collect()
    #         print("res type",type(res))
    #
    #         # res = rdd.map(tokenize_file).take(len(files))
    #
    #         # releasing memory
    #         # for (id, persistent_rdd) in sc._jsc.getPersistentRDDs().items():
    #         #     persistent_rdd.unpersist()
    #         #     print("Unpersisted {} rdd".format(id))
    #         # rdd.unpersist(blocking=True)
    #         # rdd_map.unpersist(blocking=True)
    #         # sqlContext.clearCache()
    #         # spark.
    #         # rdd.destroy()
    #         # del rdd
    #         # spark.catalog.clearCache()
    #         # sc._jvm.System.gc()
    #         # gc.collect()
    #         # spark.sparkContext.stop()
    #         # spark.stop()
    #
    #         for i in range(len(res)):
    #             token, label = res[i]
    #             for i in range(len(token)):
    #                 if token[i].dtype == tf.int32:
    #                     tokens.append(token[i])
    #                     labels.append(label[i])
    #
    #         tokens = tf.concat(tokens, axis=0)
    #         labels = tf.concat(labels, axis=0)
    #
    #         # return tokens, labels
    #         if sop or mlm:
    #             dataset = tf.data.Dataset.from_tensor_slices((tokens, labels))
    #         else:
    #             dataset = tf.data.Dataset.from_tensor_slices(tokens)
    #
    #         del tokens
    #         del labels
    #         dataset_path = res_path + r"\part_" + str(count)
    #         try:
    #             os.mkdir(dataset_path)
    #         except:
    #             print("Already exists:",dataset_path)
    #         save_labeled_dataset(dataset, dataset_path)
    #         del dataset

        # return dataset

    def tokenize_file(self, path, max_len=512, mlm=False, sop=False):
        with open(path, 'r') as file:
            text = file.read()

        tokenized_ar, label_ar = self.tokenize(text, max_len, mlm, sop)
        return tokenized_ar, label_ar

    def tokenize(self, text, max_len=512, mlm=False, sop=False):
        if sop and mlm:
            raise Exception("Select one option: mlm, sop or none of them")

        tokens = []
        segment_id = 0

        for word in text.split():
            word = word.strip()
            if word in self.vocab:
                tokens.append(self.vocab[word])
            else:
                tokens.append(self.vocab[self.unk])

        tokenized = []

        delim = self.vocab[";"]

        temp_t = []
        temp_s = []
        labels = []

        for fragm in split_list(tokens, delim):
            # new sequence if fragment is larger than max_len
            if len(fragm) >= max_len - 1:
                temp_s = []
                temp_t = []
                continue
            # + [CLS] & [SEP] & ';'
            if len(temp_t) + len(fragm) + 1 + 2 < max_len:
                if len(temp_t) != 0:
                    temp_t += [delim]
                    temp_s += [0]
                temp_t += fragm
                temp_s += [0 for _ in range(len(fragm))]
            else:
                if sop:
                    temp_s, temp_t, labels = sop_preprocess(temp_s, temp_t, labels, delim)

                if len(temp_t) > 0:
                    # add [CLS] token to start
                    temp_t = [self.vocab["[CLS]"]] + temp_t
                    temp_s = [0] + temp_s

                    # fill with [PAD] to match max_len
                    if len(temp_t) < max_len:
                        for i in range(max_len - len(temp_t)):
                            temp_t.append(self.vocab["[PAD]"])
                            temp_s.append(1 if sop else 0)

                    if mlm:
                        labels, temp_t = mlm_preprocess(labels, temp_t, self.vocab)

                    temp_t = tf.convert_to_tensor(temp_t, dtype=tf.int32)
                    temp_s = tf.convert_to_tensor(temp_s, dtype=tf.int32)
                    if len(temp_t) == max_len:
                        tokenized.append(tf.stack([temp_t, temp_s], axis=0))
                    temp_t = []
                    temp_s = []
                    temp_t += fragm
                    temp_s += [0 for _ in range(len(fragm))]
        return tokenized, labels

    def detokenize(self, tokens):
        return " ".join(self.inv_vocab[i] for i in tokens.numpy())  # tf.concat(to_str, axis=-1)


