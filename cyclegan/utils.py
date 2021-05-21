import datetime
import numpy as np
import copy
import cyclegan.write_midi as write_midi
import tensorflow as tf


# new added functions for cyclegan
class ImagePool(object):

    def __init__(self, maxsize=50):
        self.maxsize = maxsize
        self.num_img = 0
        self.images = []

    def __call__(self, image):
        if self.maxsize <= 0:
            return image
        if self.num_img < self.maxsize:
            self.images.append(image)
            self.num_img += 1
            return image
        if np.random.rand() > 0.5:
            idx = int(np.random.rand()*self.maxsize)
            tmp1 = copy.copy(self.images[idx])[0]
            self.images[idx][0] = image[0]
            idx = int(np.random.rand()*self.maxsize)
            tmp2 = copy.copy(self.images[idx])[1]
            self.images[idx][1] = image[1]
            return [tmp1, tmp2]
        else:
            return image


def load_npy_data(npy_data):
    npy_A = np.load(npy_data[0]) * 1.  # 64 * 84 * 1
    npy_B = np.load(npy_data[1]) * 1.  # 64 * 84 * 1
    npy_AB = np.concatenate((npy_A.reshape(npy_A.shape[0], npy_A.shape[1], 1),
                             npy_B.reshape(npy_B.shape[0], npy_B.shape[1], 1)),
                            axis=2)  # 64 * 84 * 2
    return npy_AB


def save_midis(bars, file_path, tempo=80.0, resolution=16):
    padded_bars = np.concatenate((np.zeros((bars.shape[0], bars.shape[1], 24, bars.shape[3])), bars,
                                  np.zeros((bars.shape[0], bars.shape[1], 20, bars.shape[3]))), axis=2)
    pause = np.zeros((bars.shape[0], 64, 128, bars.shape[3]))
    images_with_pause = padded_bars
    images_with_pause = images_with_pause.reshape(-1, 64, padded_bars.shape[2], padded_bars.shape[3])
    images_with_pause_list = []
    for ch_idx in range(padded_bars.shape[3]):
        images_with_pause_list.append(images_with_pause[:, :, :, ch_idx].reshape(images_with_pause.shape[0],
                                                                                 images_with_pause.shape[1],
                                                                                 images_with_pause.shape[2]))
    # write_midi.write_piano_rolls_to_midi(images_with_pause_list, program_nums=[33, 0, 25, 49, 0],
    #                                      is_drum=[False, True, False, False, False], filename=file_path, tempo=80.0)
    write_midi.write_piano_rolls_to_midi(images_with_pause_list, program_nums=[0], is_drum=[False], filename=file_path,
                                         tempo=tempo, beat_resolution=resolution)

# def save_midis(bars, file_path, tempo=80.0, beat_resolution=4):
#     # padded_bars = np.concatenate((np.zeros((bars.shape[0], bars.shape[1], 24, bars.shape[3])),
#     #                               bars,
#     #                               np.zeros((bars.shape[0], bars.shape[1], 20, bars.shape[3]))),
#     #                              axis=2)
#     # padded_bars = padded_bars.reshape(-1, 64, padded_bars.shape[2], padded_bars.shape[3])
#     # padded_bars_list = []
#     # for ch_idx in range(padded_bars.shape[3]):
#     #     padded_bars_list.append(padded_bars[:, :, :, ch_idx].reshape(padded_bars.shape[0],
#     #                                                                  padded_bars.shape[1],
#     #                                                                  padded_bars.shape[2]))
#     # # this is for multi-track version
#     # # write_midi.write_piano_rolls_to_midi(padded_bars_list, program_nums=[33, 0, 25, 49, 0],
#     # #                                      is_drum=[False, True, False, False, False], filename=file_path, tempo=80.0)
#     #
#     # # this is for single-track version
#     # write_midi.write_piano_rolls_to_midi(piano_rolls=padded_bars_list,
#     #                                      program_nums=[0],
#     #                                      is_drum=[False],
#     #                                      filename=file_path,
#     #                                      tempo=tempo,
#     #                                      beat_resolution=beat_resolution)
#     padded_bars = np.concatenate((np.zeros((bars.shape[0], bars.shape[1], 24, bars.shape[3])), bars,
#                                   np.zeros((bars.shape[0], bars.shape[1], 20, bars.shape[3]))), axis=2)
#     pause = np.zeros((bars.shape[0], 64, 128, bars.shape[3]))
#     images_with_pause = padded_bars
#     images_with_pause = images_with_pause.reshape(-1, 64, padded_bars.shape[2], padded_bars.shape[3])
#     images_with_pause_list = []
#     for ch_idx in range(padded_bars.shape[3]):
#         images_with_pause_list.append(images_with_pause[:, :, :, ch_idx].reshape(images_with_pause.shape[0],
#                                                                                  images_with_pause.shape[1],
#                                                                                  images_with_pause.shape[2]))
#     # write_midi.write_piano_rolls_to_midi(images_with_pause_list, program_nums=[33, 0, 25, 49, 0],
#     #                                      is_drum=[False, True, False, False, False], filename=file_path, tempo=80.0)
#     write_midi.write_piano_rolls_to_midi(images_with_pause_list, program_nums=[0], is_drum=[False], filename=file_path,
#                                          tempo=tempo, beat_resolution=beat_resolution)
#
# # def save_midis_with_params(bars, file_path, instr_idx = 0, tempo=80.0, resolution=4, duration=270, midi_info=None):
# #     padded_bars = np.concatenate((np.zeros((bars.shape[0], bars.shape[1], 24, bars.shape[3])),
# #                                   bars,
# #                                   np.zeros((bars.shape[0], bars.shape[1], 20, bars.shape[3]))),
# #                                  axis=2)
# #     padded_bars = padded_bars.reshape(-1, 64, padded_bars.shape[2], padded_bars.shape[3])
# #     padded_bars_list = []
# #     for ch_idx in range(padded_bars.shape[3]):
# #         padded_bars_list.append(padded_bars[:, :, :, ch_idx].reshape(padded_bars.shape[0],
# #                                                                      padded_bars.shape[1],
# #                                                                      padded_bars.shape[2]))
# #     # this is for multi-track version
# #     # write_midi.write_piano_rolls_to_midi(padded_bars_list, program_nums=[33, 0, 25, 49, 0],
# #     #                                      is_drum=[False, True, False, False, False], filename=file_path, tempo=80.0)
# #
# #     # this is for single-track version
# #     if type(midi_info) != None:
# #         tempo = int(midi_info['tempo'] * 0.8)
# #         resolution = midi_info['resolution'] // 32
# #         duration = midi_info['duration']
# #         print(tempo, resolution, duration)
# #     program_num = 0
# #     program_name = 'Piano'
# #     # adjust_times
# #     # from pypianoroll import Multitrack, Track, StandardTrack
# #     # tracks = []
# #     # for i in range(len(padded_bars_list)):
# #     #     track = StandardTrack(name = program_name,
# #     #                   program = program_num,
# #     #                   is_drum = program_name == "Drum",
# #     #                   pianoroll = padded_bars_list[i])
# #     #     tracks.append(track)
# #     #
# #     # multitrack = Multitrack(resolution = resolution,
# #     #                         tempo = np.array([tempo] * tracks[0].get_length()),
# #     #                         tracks = tracks)
# #     # multitrack = multitrack.set_resolution(resolution)
# #     # multitrack = multitrack.se
# #     # pm = multitrack.to_pretty_midi()
# #     # pm.adjust_times(pm.get_end_time(), duration)
# #     # pm.write(file_path)
# #
# #     # cur_duration = pm.e
# #     write_midi.write_piano_rolls_to_midi(piano_rolls=padded_bars_list,
# #                                          program_nums=[program_num],
# #                                          is_drum=[False],
# #                                          filename=file_path,
# #                                          tempo=tempo,
# #                                          beat_resolution=resolution)
# def save_midis_params(bars, file_path, instr_idx = 0, tempo=80.0, resolution=4, duration=270, midi_info=None):
#     padded_bars = np.concatenate((np.zeros((bars.shape[0], bars.shape[1], 24, bars.shape[3])),
#                                   bars,
#                                   np.zeros((bars.shape[0], bars.shape[1], 20, bars.shape[3]))),
#                                  axis=2)
#     padded_bars = padded_bars.reshape(-1, 64, padded_bars.shape[2], padded_bars.shape[3])
#     padded_bars_list = []
#     for ch_idx in range(padded_bars.shape[3]):
#         padded_bars_list.append(padded_bars[:, :, :, ch_idx].reshape(padded_bars.shape[0],
#                                                                      padded_bars.shape[1],
#                                                                      padded_bars.shape[2]))
#
#     # padded_bars = padded_bars.reshape()
#     # padded_bars_list = []
#     # for ch_idx in range(padded_bars.shape[3]):
#     #     padded_bars_list.append(padded_bars[:, :, :, ch_idx].reshape(padded_bars.shape[0],
#     #                                                                  padded_bars.shape[1],
#     #                                                                  padded_bars.shape[2]))
#
#     # this is for multi-track version
#     # write_midi.write_piano_rolls_to_midi(padded_bars_list, program_nums=[33, 0, 25, 49, 0],
#     #                                      is_drum=[False, True, False, False, False], filename=file_path, tempo=80.0)
#
#     # this is for single-track version
#     if midi_info is not None:
#         tempo = int(midi_info['tempo'] * 0.8)
#         resolution = midi_info['resolution'] // 32
#         duration = midi_info['duration']
#         print(tempo, resolution, duration)
#     program_num = 0
#     program_name = 'Piano'
#     # adjust_times
#     # from pypianoroll import Multitrack, Track, StandardTrack
#     # tracks = []
#     # for i in range(len(padded_bars_list)):
#     #     track = StandardTrack(name = program_name,
#     #                   program = program_num,
#     #                   is_drum = program_name == "Drum",
#     #                   pianoroll = padded_bars_list[i])
#     #     tracks.append(track)
#     #
#     # multitrack = Multitrack(resolution = resolution,
#     #                         tempo = np.array([tempo] * tracks[0].get_length()),
#     #                         tracks = tracks)
#     # multitrack = multitrack.set_resolution(resolution)
#     # multitrack = multitrack.se
#     # pm = multitrack.to_pretty_midi()
#     # pm.adjust_times(pm.get_end_time(), duration)
#     # pm.write(file_path)
#
#     # cur_duration = pm.e
#     write_midi.write_piano_rolls_to_midi(piano_rolls=padded_bars_list,
#                                          program_nums=[program_num],
#                                          is_drum=[False],
#                                          filename=file_path,
#                                          tempo=tempo,
#                                          beat_resolution=resolution)


def get_now_datetime():
    now = datetime.datetime.now().strftime('%Y-%m-%d')
    return str(now)


def to_binary(bars, threshold=0.0):
    """Turn velocity value into boolean"""
    track_is_max = tf.equal(bars, tf.reduce_max(bars, axis=-1, keepdims=True))
    track_pass_threshold = (bars > threshold)
    out_track = tf.logical_and(track_is_max, track_pass_threshold)
    return out_track
