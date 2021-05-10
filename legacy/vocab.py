from pathlib import Path
import json
from tqdm import tqdm


def build_vocab(
        texts_dirs=[r"D:\DIPLOMA\dataset\big midi\nesmdb\texts"],
        output_dir=r"D:\DIPLOMA\dataset\big midi\nesmdb"
):
    paths = []
    for texts_dir in texts_dirs:
        paths += [str(x) for x in Path(texts_dir).glob("*.txt")]
    vocab = {}

    # get all words
    for path in tqdm(paths):
        with open(path, 'r') as file:
            text = file.read()
            for word in text.split():
                word = word.strip()
                if word in vocab:
                    vocab[word] += 1
                else:
                    vocab[word] = 1
    ordered_vocab = [(key, vocab[key]) for key in vocab.keys()]
    ordered_vocab = sorted(ordered_vocab, key=lambda x: -x[1])

    vocab = {
        "[CLS]": 0,
        "[SEP]": 1,
        "[PAD]": 2,
        "[UNK]": 3,
        "[MASK]": 4
    }

    for key, _ in ordered_vocab:
        vocab[key] = len(vocab)

    if output_dir:
        with open(output_dir + '\\vocab.json', 'w') as file:
            json.dump(vocab, file)

    return vocab


def load_vocab(path=r"D:\DIPLOMA\dataset\big midi\nesmdb\vocab.json"):
    with open(path, 'r') as file:
        text = file.read()
        vocab = json.loads(text)
        return vocab

