import tensorflow as tf
import os
from multiprocessing import Pool
# from billiard.pool import Pool
import multiprocessing
import numpy as np
from legacy.multiprocessing_file_defs import tokenize_file


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


def tokenize_part(data):
    vocab = data[0]
    paths = data[1]
    max_len = data[2]
    mlm = data[3]
    sop = data[4]
    batch_size = data[5]
    count = data[6]
    res_path = data[7]

    tokens = []
    labels = []

    # count += 1
    r = range(len(paths))
    paths = list([paths[r:r + batch_size] for r in r[::batch_size]])

    files = [(vocab, path, max_len, mlm, sop) for path in paths]
    del paths

    cpu_count = multiprocessing.cpu_count() // 2

    print("Group", count, ": consists of", batch_size, "in each batch, on", cpu_count, "CPUs")

    with Pool(cpu_count) as p:
        res = p.map(tokenize_file, files)
    del files
    print("res type", type(res))

    for i in range(len(res)):
        token, label = res[i]
        for i in range(len(token)):
            if token[i].dtype == tf.int32:
                tokens.append(token[i])
                labels.append(label[i])

    tokens = tf.concat(tokens, axis=0)
    labels = tf.concat(labels, axis=0)

    # return tokens, labels
    if sop or mlm:
        dataset = tf.data.Dataset.from_tensor_slices((tokens, labels))
    else:
        dataset = tf.data.Dataset.from_tensor_slices(tokens)

    del tokens
    del labels
    dataset_path = res_path + r"\part_" + str(count)
    try:
        os.mkdir(dataset_path)
    except:
        print("Already exists:", dataset_path)

    if sop or mlm:
        save_labeled_dataset(dataset, dataset_path)
    else:
        tf.data.experimental.save(dataset, dataset_path)
    del dataset
    return (0)

