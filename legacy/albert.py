import tensorflow as tf
devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(devices[0], True)
import tensorflow_addons as tfa
from bert import BertModelLayer
from tensorflow import keras
from legacy.vocab import load_vocab, build_vocab
from legacy.dataset import load_labeled_dataset
from legacy.transcription import decode
from tqdm import tqdm
import os
import math
from random import shuffle
import datetime


def make_bert(num_tokens, H, A, L, V, E, feed_forward_size):
    input_ids = keras.layers.Input(shape=(num_tokens,), dtype='int32')
    token_type_ids = keras.layers.Input(shape=(num_tokens,), dtype='int32')

    albert = BertModelLayer(**BertModelLayer.Params(
        vocab_size=V,  # embedding params
        use_token_type=True,
        use_position_embeddings=True,
        max_position_embeddings=num_tokens,
        token_type_vocab_size=2,

        num_heads=A,
        num_layers=L,  # transformer encoder params
        hidden_size=H,
        hidden_dropout=0.1,
        intermediate_size=feed_forward_size,
        intermediate_activation="gelu",

        adapter_size=None,  # see arXiv:1902.00751 (adapter-BERT)

        shared_layer=True,  # True for ALBERT (arXiv:1909.11942)
        embedding_size=E  # ,         # None for BERT, wordpiece embedding size for ALBERT
    ))

    logits = albert([input_ids, token_type_ids])

    x = tf.keras.layers.Dense(V // 2, activation='relu')(logits[:, 0])
    sop_logits = tf.keras.layers.Dense(2, activation='softmax')(x)

    z = tf.keras.layers.Dense(feed_forward_size, activation='relu')(logits)

    mlm_logits = tf.keras.layers.Dense(V, activation='softmax')(z)  # (z)

    print(albert.get_config())

    model = keras.Model(inputs=[input_ids, token_type_ids], outputs=[logits, sop_logits, mlm_logits])
    model.build(input_shape=[(None, num_tokens), (None, num_tokens)])

    print(model.summary())

    return model


def fit(model, V,
        EPOCH=1,
        n_parts=1,
        batch_size=50,
        lrate=1e-3,  # 0.00176
        log_dir=r"D:\DIPLOMA\model\logs\sop",
        sop_ds_folders=r"D:\DIPLOMA\dataset\big midi\tfds\sop_dataset",
        mlm_ds_folders=r"D:\DIPLOMA\dataset\big midi\tfds\mlm_dataset"
        ):
    summary_writer = tf.summary.create_file_writer(
        log_dir + r"\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )

    tf.get_logger().setLevel('ERROR')

    accuracy = tf.keras.metrics.Accuracy()

    # to calculate loss for sop and mlm
    sop_bce = tf.keras.losses.CategoricalCrossentropy()
    mlm_bce = tf.keras.losses.SparseCategoricalCrossentropy()

    # recomeded optimizer
    opt = tfa.optimizers.LAMB(learning_rate=lrate)
    # opt = tf.keras.optimizers.SGD(learning_rate=lrate)

    for epoch in range(EPOCH):
        print("Epoch", epoch + 1)

        # get all dataset directories and calculate
        # number of dataset per part
        # datasets go in random order
        sop_dirs = os.listdir(sop_ds_folders)
        sop_n = math.ceil(len(sop_dirs) / n_parts)
        shuffle(sop_dirs)

        # same for mlm
        mlm_dirs = os.listdir(mlm_ds_folders)
        mlm_n = math.ceil(len(mlm_dirs) / n_parts)
        shuffle(mlm_dirs)

        for i in range(sop_n + mlm_n):
            dataset = None
            # chose objective for this iteration
            is_sop = i % 2 == 1

            # get directories for current part
            if is_sop:
                print("SOP task")
                cur_dirs = sop_dirs[:n_parts]
                sop_dirs = sop_dirs[n_parts::]
            else:
                print("MLM task")
                cur_dirs = mlm_dirs[:n_parts]
                mlm_dirs = mlm_dirs[n_parts::]

            print(i + 1, "part out of", sop_n + mlm_n)
            print(cur_dirs)

            # concatenate datasets for current part
            for folder in cur_dirs:
                if is_sop:
                    ds = sop_ds_folders + "\\" + folder
                else:
                    ds = mlm_ds_folders + "\\" + folder
                print(ds)
                if dataset:
                    dataset = dataset.concatenate(load_labeled_dataset(ds))
                else:
                    dataset = load_labeled_dataset(ds)

            progress_bar = tqdm(dataset.batch(batch_size))
            with tf.device('GPU:0'):
                for batch in progress_bar:
                    # unpack batch
                    sentences, label = batch
                    token = sentences[:, 0]
                    segment = sentences[:, 1]
                    # learning step
                    with tf.GradientTape(persistent=True) as tape:
                        output, cls, token_probs = model([token, segment])
                        if is_sop:
                            loss = sop_bce(label, cls)
                            acc = accuracy(
                                tf.math.argmax(label, 1),
                                tf.math.argmax(cls, 1)
                            )
                        else:
                            one_hot_label = tf.one_hot(label, V)
                            loss = mlm_bce(label, token_probs)
                            predicted_labels = tf.math.argmax(token_probs, 2)
                            acc = accuracy(label, predicted_labels)
                    # updating model weights
                    grad = tape.gradient(loss, model.trainable_variables)
                    opt.apply_gradients(zip(grad, model.trainable_variables))
                    progress_bar.set_description(f"loss = {loss.numpy()} acc = {acc.numpy()}")
            print("Part", i, "end\n")
        print("Epoch", epoch, "end\n")
    print("End of training")
    tf.get_logger().setLevel('INFO')


def make_decoder_model(trained_model, num_tokens, H, feed_forward_size, V):
    logits = keras.layers.Input(shape=(num_tokens, H), dtype='int32')
    z = tf.keras.layers.Dense(feed_forward_size, activation='relu')(logits)
    res = tf.keras.layers.Dense(V, activation='softmax')(z)
    model = keras.Model(inputs=logits, outputs=res)
    model.compile()

    trained_weights = [trained_model.get_weights()[-6]] + [
        trained_model.get_weights()[-5]] + trained_model.get_weights()[-2:]
    model.set_weights(trained_weights)
    return model


def emb_to_text(emb, decoder_model, tokenizer):
    tokens = tf.math.argmax(decoder_model(emb), axis=2)
    return [tokenizer.detokenize(tokens[i]) for i in range(tokens.shape[0])]


def to_midi(texts, OG_res=480, OG_tempo=296):
    return [decode(text, OG_res, OG_tempo) for text in texts]


def main():

    output_dir = r"D:\DIPLOMA\dataset"
    vocab_dir = output_dir + r"\vocab.json"

    # vocab = build_vocab(texts_dirs, output_dir)
    vocab = load_vocab(vocab_dir)
    vocab_file = r"D:\DIPLOMA\dataset\vocab.json"

    # without mlm or sop
    sop_ds = r"D:/DIPLOMA/dataset/big midi/tfds/sop_dataset"
    mlm_ds = r"D:/DIPLOMA/dataset/big midi/tfds/mlm_dataset"

    vocab_file = r"D:\DIPLOMA\dataset\vocab.json"
    max_len = 128

    vocab = load_vocab(vocab_file)

    H = 64  # 1024 # H >> E
    A = H // 64  # num self attention heads
    L = 6  # 12
    V = len(vocab)  # vocab size
    E = 32  # 256 # vocabluary embedding size
    feed_forward_size = H  # 4 * H
    num_tokens = max_len

    # model = make_bert(num_tokens, H, A, L, V, E, feed_forward_size)

    save_path = r"D:\DIPLOMA\model\saved"
    model = tf.keras.models.load_model(save_path)

    tf.keras.utils.plot_model(model, "albert.png", show_shapes=True)

    devices = tf.config.list_physical_devices('GPU')
    # tf.config.experimental.set_memory_growth(devices[0], True)
    devices[0]

    fit(model, V, sop_ds_folders=sop_ds, mlm_ds_folders=mlm_ds)

    save_path = r"D:\DIPLOMA\model\saved"
    model.save(save_path)

    decoder_model = make_decoder_model(model, num_tokens, H, feed_forward_size, V)
    return model, decoder_model

