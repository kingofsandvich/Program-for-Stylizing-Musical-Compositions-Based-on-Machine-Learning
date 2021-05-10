import tensorflow as tf
import os
from legacy.tokenizer import MusicTokenizer


def tokenize_file(data):
    vocab = data[0]
    paths = data[1]
    max_len = data[2]
    mlm = data[3]
    sop = data[4]

    tokens = []
    segments = []

    tokenizer = MusicTokenizer(vocab)

    for path in paths:
        size = os.path.getsize(path)
        max_Mb = 0.4
        Mb = 2**20
        if size <= Mb * max_Mb:
            # try:
            token, segment = tokenizer.tokenize_file(path, max_len, mlm, sop)

            token = tf.stack(token, axis=0)
            segment = tf.stack(segment, axis=0)
            #     if len(token) == max_len:
            tokens.append(token)
            segments.append(segment)

            del token, segment
            # except Exception as e:
            #     print(e,'\n',path,"SEEMS TO BE TOO BIG", size)
        else:
            print(path,"(",size, ") is too large", size <= Mb * max_Mb, Mb * max_Mb)
    del tokenizer
    del vocab, paths, max_len, mlm, sop
    del data
    return (tokens, segments)


