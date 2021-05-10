import os
from pyspark.sql import SparkSession
import pyspark
import multiprocessing
import pretty_midi
from legacy.transcription import encode
import pandas as pd

# Functions for transcripting dataset stored in midi files using spark
# Load midi from 'processed_dir'
# Save resulted transcriptions to 'processed_dir'


# Names of particular midi files in given directory
def get_filenames(dataset_path=r'D:\ВКР\dataset\big midi\Lakh MIDI\lmd_full', ext=None):
    # count = 0
    filenames = []

    # dirs = [ name for name in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, name))]
    for file_group in os.listdir(dataset_path):
        item_name = os.path.join(dataset_path, file_group)

        # for filename in os.listdir(group_path):
        #     item_name = os.path.join(group_path, filename)
        if os.path.isdir(item_name):
            filenames += get_filenames(item_name, ext)
        else:
            if ext:
                for e in ext:
                    if item_name.endswith(e):
                        filenames.append(item_name)
                        break
            else:
                filenames.append(item_name)

    print(len(filenames), "midi files in", dataset_path)
    return filenames


# Read midi from file, make transcription and write result to 'processed_dir'
def file_preproccess(data):
    filename, processed_dir, texts_dir = data
    try:
        pm = pretty_midi.PrettyMIDI(filename)
        encoded, OG_res, OG_tempo = encode(pm)
        if processed_dir:
            procesed_file = processed_dir + "\\" + filename.split("\\")[-1].split('.')[0] + '.txt'
            with open(procesed_file, "w") as text_file:
                text_file.write(str([encoded, OG_res, OG_tempo]))
        if texts_dir:
            procesed_file = texts_dir + "\\" + filename.split("\\")[-1].split('.')[0] + '.txt'
            with open(procesed_file, "w") as text_file:
                text_file.write(str(encoded))
        return True
    except:
        return False  # str(list())


# Write transcripted files to 'processed_dir'
def preproccess_files(filenames,
                      processed_dir,
                      texts_dir,
                      spark,
                      n=-1,
                      cpu_count=4,
                      return_value=False):
    num = len(filenames) if n == -1 else n

    files = list(filenames[:num])
    files = list([(f, processed_dir, texts_dir) for f in files])
    print('cpu_count',cpu_count)
    rdd = spark.sparkContext.parallelize(files, cpu_count)
    if return_value:
        return rdd.map(file_preproccess).collect()
    rdd.map(file_preproccess).collect()


def spark_preprocess(num=-1,
                     dataset_dir=r'D:\ВКР\dataset\big midi\Lakh MIDI\lmd_full',
                     processed_dir=r"D:\ВКР\dataset\big midi\ProcessedLakh",
                     texts_dir=r'D:\ВКР\dataset\big midi\ProcessedLakh'):
    cpu_count = multiprocessing.cpu_count()

    spark = SparkSession.builder\
        .master("local[8]")\
        .appName("SparkApp")\
        .getOrCreate()
    sc = spark.sparkContext

    conf = pyspark.SparkConf().setAll([
        ('spark.executor.cores', str(cpu_count)),
        ('spark.cores.max', str(cpu_count))])
    spark.sparkContext.stop()
    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    sc = spark.sparkContext

    print(sc.getConf().getAll())

    filenames = get_filenames(dataset_path=dataset_dir, ext=[".mid", ".midi"])
    # filenames += get_filenames(dataset_path=dataset_dir, ext=)
    preproccess_files(filenames, processed_dir, texts_dir, spark, num, cpu_count)


# Save DataFrames with processed midi to csv files
def to_csv(csv_dir=r"D:\ВКР\dataset\big midi\pdLakh" + "\\",
           processed_dir=r'D:\ВКР\dataset\big midi\ProcessedLakh',
           df_size=10 * 1000):
    count = 0
    file_num = 0

    filenames_processed = get_filenames(dataset_path=processed_dir)

    for i, filename in enumerate(filenames_processed):
        if count == 0:
            df = pd.DataFrame({'title': pd.Series([], dtype='str'),
                               'encoded': pd.Series([]),
                               'resolution': pd.Series([], dtype='int'),
                               'tempo': pd.Series([], dtype='float'),
                               })

        with open(filename, 'r') as file:
            text = file.read().split("'")
            encoded = text[1]
            original = text[2][:-1].split(',')
            og_res = int(original[1])
            og_tempo = float(original[2])
            title = filename.split('\\')[-1][:-4]
            df.loc[i] = [title, encoded.split(), og_res, og_tempo]
            count = (count + 1) % df_size

        if (count == 0):
            df.to_csv(csv_dir + str(file_num) + '.csv')
            file_num += 1
    df.to_csv(csv_dir + str(file_num) + '.csv')