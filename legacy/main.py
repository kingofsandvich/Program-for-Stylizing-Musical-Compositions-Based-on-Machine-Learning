from legacy.preprocess_midi import spark_preprocess
from legacy.dataset import make_dataset, load_labeled_dataset, save_labeled_dataset
from legacy.cyclegan import make_cycle_gan, get_dataset, fit, plot_sample, save_cycle_gan
import tensorflow as tf

def prepare_dataset():
    processed_dir = None
    dataset_dir = r"D:\DIPLOMA\dataset\genre\dss\pop"
    texts_dir = r"D:\DIPLOMA\dataset\genre\texts\pop"
    spark_preprocess(num=-1,
                     dataset_dir=dataset_dir,
                     processed_dir=processed_dir,
                     texts_dir=texts_dir)

    dataset_dir = r"D:\DIPLOMA\dataset\genre\dss\classic"
    texts_dir = r"D:\DIPLOMA\dataset\genre\texts\classic"
    spark_preprocess(num=-1,
                     dataset_dir=dataset_dir,
                     processed_dir=processed_dir,
                     texts_dir=texts_dir)

    dataset_dir = r"D:\DIPLOMA\dataset\genre\dss\jazz"
    texts_dir = r"D:\DIPLOMA\dataset\genre\texts\jazz"
    spark_preprocess(num=-1,
                     dataset_dir=dataset_dir,
                     processed_dir=processed_dir,
                     texts_dir=texts_dir)

    dataset_dir = r"D:\DIPLOMA\dataset\genre\dss\rock"
    texts_dir = r"D:\DIPLOMA\dataset\genre\texts\rock"
    spark_preprocess(num=-1,
                     dataset_dir=dataset_dir,
                     processed_dir=processed_dir,
                     texts_dir=texts_dir)

    vocab_file = r"D:\DIPLOMA\dataset\vocab.json"

    max_len = 128

    pop_dir = [r"D:\DIPLOMA\dataset\genre\texts\pop"]
    rock_dir = [r"D:\DIPLOMA\dataset\genre\texts\rock"]
    classic_dir = [r"D:\DIPLOMA\dataset\genre\texts\classic"]
    jazz_dir = [r"D:\DIPLOMA\dataset\genre\texts\jazz"]

    pop_ds = r"D:\DIPLOMA\dataset\genre\tfds\pop"
    rock_ds = r"D:\DIPLOMA\dataset\genre\tfds\rock"
    classic_ds = r"D:\DIPLOMA\dataset\genre\tfds\classic"
    jazz_ds = r"D:\DIPLOMA\dataset\genre\tfds\jazz"

    make_dataset(res_path=pop_ds,
                 nparts=1,
                 vocab_file=vocab_file,
                 data_dirs=pop_dir,
                 max_len=max_len,
                 batch_size=150,
                 sop=False,
                 mlm=False)

    make_dataset(res_path=classic_ds,
                 nparts=1,
                 vocab_file=vocab_file,
                 data_dirs=classic_dir,
                 max_len=max_len,
                 batch_size=150,
                 sop=False,
                 mlm=False)

    make_dataset(res_path=rock_ds,
                 nparts=1,
                 vocab_file=vocab_file,
                 data_dirs=rock_dir,
                 max_len=max_len,
                 batch_size=150,
                 sop=False,
                 mlm=False)

    make_dataset(res_path=jazz_ds,
                 nparts=1,
                 vocab_file=vocab_file,
                 data_dirs=jazz_dir,
                 max_len=max_len,
                 batch_size=150,
                 sop=False,
                 mlm=False)


def train_cyclegan():
    # trained CycleGAN location
    pj_cgan_path = r"D:\DIPLOMA\CycleGANs\pop_jazz"
    cj_cgan_path = r"D:\DIPLOMA\CycleGANs\classic_jazz"

    # processed dataset location
    pj_dataset_path = r"D:\DIPLOMA\dataset\genre\tfds\pop_jazz"
    cj_dataset_path = r"D:\DIPLOMA\dataset\genre\tfds\classic_jazz"

    pj_cache = r"D:\DIPLOMA\dataset\genre\cache\pop_jazz\\"
    cj_cache = r"D:\DIPLOMA\dataset\genre\cache\classic_jazz\\"

    # trainde ALBERT model
    embedding_path = r"D:\DIPLOMA\model\saved"

    # saved tf.Datasets [None, NUM_TOKENS, EMB_SIZE]
    classic_dir = r"D:\DIPLOMA\dataset\genre\tfds\classic\part_1"
    jazz_dir = r"D:\DIPLOMA\dataset\genre\tfds\jazz\part_1"
    pop_dir = r"D:\DIPLOMA\dataset\genre\tfds\pop\part_1"
    rock_dir = r"D:\DIPLOMA\dataset\genre\tfds\rock\part_1"

    NUM_TOKENS = 128
    EMB_SIZE = 64

    # Load or create CycleGAN model
    gen_g, gen_f, discr_x, discr_y = make_cycle_gan(NUM_TOKENS, EMB_SIZE,
                                                    16, filter_size=2)
    print(gen_g.summary())
    print(discr_x.summary())

    dataset = get_dataset(cj_dataset_path, cj_cache)

    # plot_sample(dataset, gen_g, gen_f)
    # print(tf.test.is_gpu_available())
    # print(tf.test.is_built_with_cuda())
    fit(dataset, gen_g, gen_f, discr_x, discr_y, n_epoch=10,
        lrate=2e-4, prob=0.2, batch=50)
    plot_sample(dataset, gen_g, gen_f)

    save_cycle_gan(gen_g, gen_f, discr_x, discr_y, cj_cgan_path)


def main():
    train_cyclegan()


if __name__ == '__main__':
    main()


