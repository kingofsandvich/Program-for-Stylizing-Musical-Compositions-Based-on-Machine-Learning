# from .. model import CycleGAN
# from .. main import args

from cyclegan.model import CycleGAN
import argparse
import os, shutil
import binascii
from pathlib import Path


def make_args():
    parser = argparse.ArgumentParser(description='')
    # Dataset
    parser.add_argument('--dataset_A_dir', dest='dataset_A_dir', default='CP_C', help='path of the dataset of domain A')
    parser.add_argument('--dataset_B_dir', dest='dataset_B_dir', default='CP_P', help='path of the dataset of domain B')

    # Training hyperparams
    parser.add_argument('--epoch', dest='epoch', type=int, default=6, help='# of epoch')
    parser.add_argument('--epoch_step', dest='epoch_step', type=int, default=10, help='# of epoch to decay lr')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=4, help='# images in batch')
    parser.add_argument('--time_step', dest='time_step', type=int, default=64, help='time step of pianoroll')
    parser.add_argument('--L1_lambda', dest='L1_lambda', type=float, default=10.0, help='weight on L1 term in objective')
    parser.add_argument('--gamma', dest='gamma', type=float, default=1.0, help='weight of extra discriminators')
    parser.add_argument('--max_size', dest='max_size', type=int, default=50, help='max size of image pool, 0 means do not use image pool')
    parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')
    parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate for adam')
    parser.add_argument('--print_freq', dest='print_freq', type=int, default=100, help='print the debug information every print_freq iterations')
    parser.add_argument('--sigma_d', dest='sigma_d', type=float, default=0.01, help='sigma of gaussian noise of discriminators')

    # Data configuration
    parser.add_argument('--pitch_range', dest='pitch_range', type=int, default=84, help='pitch range of pianoroll')
    parser.add_argument('--input_nc', dest='input_nc', type=int, default=1, help='# of input image channels')
    parser.add_argument('--output_nc', dest='output_nc', type=int, default=1, help='# of output image channels')

    # Paths configuration
    parser.add_argument('--test_dir', dest='test_dir', default='./test', help='test sample are saved here')
    parser.add_argument('--res_dir', dest='res_dir', default='./results', help='results for genre transfer in prod phase goes here')
    parser.add_argument('--source_dir', dest='source_dir', default='./sources', help='source of midi files for prod phase')
    parser.add_argument('--log_dir', dest='log_dir', default='./log', help='logs are saved here')
    parser.add_argument('--sigma_c', dest='sigma_c', type=float, default=0.01, help='sigma of gaussian noise of classifiers')
    parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint', help='models are saved here')
    parser.add_argument('--checkpoint_name', dest='checkpoint_name', default='CycleGAN_CP', help='name of the checkpoint')
    parser.add_argument('--sample_dir', dest='sample_dir', default='./samples', help='sample are saved here')
    parser.add_argument('--save_freq', dest='save_freq', type=int, default=1000, help='save a model every save_freq iterations')

    # Model configuration
    parser.add_argument('--ngf', dest='ngf', type=int, default=64, help='# of gen filters in first conv layer')
    parser.add_argument('--ndf', dest='ndf', type=int, default=64, help='# of discri filters in first conv layer')
    # for test
    parser.add_argument('--which_direction', dest='which_direction', default='AtoB', help='AtoB or BtoA')
    # continue_train
    parser.add_argument('--continue_train', dest='continue_train', type=bool, default=True, help='if continue training, load the latest model: 1: true, 0: false')
    # retry with full
    parser.add_argument('--model', dest='model', default='full', help='three different models, base, partial, full')
    parser.add_argument('--type', dest='type', default='cyclegan', help='cyclegan or classifier')

    # Other
    parser.add_argument('--phase', dest='phase', default='prod', help='train, test, prod')

    args = parser.parse_args()
    return args


def init_model(model_name, atob):
    args = make_args()
    args.phase = "prod"
    args.checkpoint_name = model_name
    args.which_direction = "AtoB" if atob else "BtoA"

    cur_path = Path(os.getcwd())
    # cur_path = cur_path.parent.absolute()
    args.checkpoint_dir = os.path.join(cur_path, "checkpoint")

    model = CycleGAN(args)

    return model, args


def clear_folder(folder):
    for filename in os.listdir(folder):
        if os.path.isfile(os.path.join(folder, filename)):
            os.remove(os.path.join(folder, filename))


def run_model(model, midi_file, args):
    # ensure needed directories exist and are empty
    if not os.path.exists(args.checkpoint_dir):
        raise Exception("Can not find checkpoint directory.")

    if not os.path.exists(args.source_dir):
        os.makedirs(args.source_dir)

    if not os.path.exists(args.res_dir):
        os.makedirs(args.res_dir)

    clear_folder(args.source_dir)
    clear_folder(args.res_dir)

    # write received file to source directory
    with open(os.path.join(args.source_dir, "tmp.mid"), "wb") as file:
        file.write(binascii.unhexlify(midi_file))

    # run the model
    path = model.prod(args)

    if path:
        # read translated file and return as string
        with open(path, "rb") as file:
            res = binascii.hexlify(file.read())#.decode()
    else:
        raise Exception("Could not read resulted midi.")

    return res

