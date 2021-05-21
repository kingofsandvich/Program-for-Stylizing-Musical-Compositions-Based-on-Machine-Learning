import os
import time
from glob import glob
import numpy as np
from collections import namedtuple
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from cyclegan.module import build_generator, build_discriminator, abs_criterion, mae_criterion
from cyclegan.utils import get_now_datetime, ImagePool, to_binary, load_npy_data, save_midis#, save_midis_params


class CycleGAN(object):

    def __init__(self, args):

        self.batch_size = args.batch_size
        self.time_step = args.time_step  # number of time steps
        self.pitch_range = args.pitch_range  # number of pitches
        self.input_c_dim = args.input_nc  # number of input image channels
        self.output_c_dim = args.output_nc  # number of output image channels
        self.lr = args.lr
        self.L1_lambda = args.L1_lambda
        self.gamma = args.gamma
        self.sigma_d = args.sigma_d
        self.dataset_A_dir = args.dataset_A_dir
        self.dataset_B_dir = args.dataset_B_dir
        self.source_dir = args.source_dir
        self.res_dir = args.res_dir
        self.sample_dir = args.sample_dir

        self.model = args.model
        self.discriminator = build_discriminator
        self.generator = build_generator
        self.criterionGAN = mae_criterion

        OPTIONS = namedtuple('OPTIONS', 'batch_size '
                                        'time_step '
                                        'input_nc '
                                        'output_nc '
                                        'pitch_range '
                                        'gf_dim '
                                        'df_dim '
                                        'is_training')
        self.options = OPTIONS._make((args.batch_size,
                                      args.time_step,
                                      args.input_nc,
                                      args.output_nc,
                                      args.pitch_range,
                                      args.ngf,
                                      args.ndf,
                                      args.phase == 'train'))

        self.now_datetime = get_now_datetime()
        self.pool = ImagePool(args.max_size)

        self._build_model(args)

        print("initialize model...")

    def _build_model(self, args):
        # print(self.options)
        # Generator
        self.generator_A2B = self.generator(self.options,
                                            name='Generator_A2B')
        self.generator_B2A = self.generator(self.options,
                                            name='Generator_B2A')
        # print(self.generator_A2B.summary())

        # Discriminator
        self.discriminator_A = self.discriminator(self.options,
                                                  name='Discriminator_A')
        self.discriminator_B = self.discriminator(self.options,
                                                  name='Discriminator_B')

        if self.model != 'base':
            self.discriminator_A_all = self.discriminator(self.options,
                                                          name='Discriminator_A_all')
            self.discriminator_B_all = self.discriminator(self.options,
                                                          name='Discriminator_B_all')

        # Discriminator and Generator Optimizer
        self.DA_optimizer = Adam(self.lr,
                                 beta_1=args.beta1)
        self.DB_optimizer = Adam(self.lr,
                                 beta_1=args.beta1)
        self.GA2B_optimizer = Adam(self.lr,
                                   beta_1=args.beta1)
        self.GB2A_optimizer = Adam(self.lr,
                                   beta_1=args.beta1)

        if self.model != 'base':
            self.DA_all_optimizer = Adam(self.lr,
                                         beta_1=args.beta1)
            self.DB_all_optimizer = Adam(self.lr,
                                         beta_1=args.beta1)

        # Checkpoints
        model_name = "cyclegan.model"
        model_dir = "{}2{}_{}_{}_{}".format(self.dataset_A_dir,
                                            self.dataset_B_dir,
                                            self.now_datetime,
                                            self.model,
                                            self.sigma_d)
        model_dir = args.checkpoint_name
        self.checkpoint_dir = os.path.join(args.checkpoint_dir,
                                           model_dir,
                                           model_name)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        if self.model == 'base':
            self.checkpoint = tf.train.Checkpoint(generator_A2B_optimizer=self.GA2B_optimizer,
                                                  generator_B2A_optimizer=self.GB2A_optimizer,
                                                  discriminator_A_optimizer=self.DA_optimizer,
                                                  discriminator_B_optimizer=self.DB_optimizer,
                                                  generator_A2B=self.generator_A2B,
                                                  generator_B2A=self.generator_B2A,
                                                  discriminator_A=self.discriminator_A,
                                                  discriminator_B=self.discriminator_B)
        else:
            self.checkpoint = tf.train.Checkpoint(generator_A2B_optimizer=self.GA2B_optimizer,
                                                  generator_B2A_optimizer=self.GB2A_optimizer,
                                                  discriminator_A_optimizer=self.DA_optimizer,
                                                  discriminator_B_optimizer=self.DB_optimizer,
                                                  discriminator_A_all_optimizer=self.DA_all_optimizer,
                                                  discriminator_B_all_optimizer=self.DB_all_optimizer,
                                                  generator_A2B=self.generator_A2B,
                                                  generator_B2A=self.generator_B2A,
                                                  discriminator_A=self.discriminator_A,
                                                  discriminator_B=self.discriminator_B,
                                                  discriminator_A_all=self.discriminator_A_all,
                                                  discriminator_B_all=self.discriminator_B_all)

        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint,
                                                             self.checkpoint_dir,
                                                             max_to_keep=5)

        # if self.checkpoint_manager.latest_checkpoint:
        #     self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
        #     print('Latest checkpoint restored!!')

    def prod(self, args):
        from legacy.preprocess_midi import get_filenames
        from convert_clean import convert
        from math import ceil

        print("Produce genre transfered midi from midi file.")

        if self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint):
            print(" [*] Load checkpoint succeeded!"*100)
        else:
            print(" [!] Load checkpoint failed...")

        source = get_filenames(self.source_dir, ext=['midi', 'mid'])#glob('{}/*.*'.format(self.source_dir))

        idxs = range(len(source))

        for idx in idxs:

            print(idx + 1, source[idx], sep='. ')

            # convert midi to proper numpy array
            res = convert(source[idx])
            if res:
                source_npy, midi_name, midi_info = res
                print(source_npy)
                # take piano
                source_npy = source_npy.stack()

                print("source_npy", source_npy.shape)

                # for i in range(n_batches):
                instr_num = 4
                instr_num = np.argmax(source_npy.sum(axis=1).sum(axis=1), axis=0)
                # print('instr_num', np.argmax(source_npy.sum(axis=1).sum(axis=1), axis=0))
                source_npy = source_npy[instr_num]

                # take only part of pitch range
                start_range = 20
                source_npy = source_npy[:,start_range:start_range + args.pitch_range:]
                num_batches = ceil(source_npy.shape[0] / args.time_step)
                zeros = np.zeros([num_batches * args.time_step, source_npy.shape[1]])
                # complete with zeros to match size (num_batches * time_step)
                # print(zeros[:source_npy.shape[0]:,:].shape)
                zeros[:source_npy.shape[0]:,:] = source_npy
                source_npy = zeros
                del zeros
                print(type(source_npy), source_npy.shape)
                print("Must be: (1, 64, 84, 1) <class 'numpy.ndarray'>")


                # source_npy *= 1
                # sample_npy = np.load(sample_files[idx]) * 1.

                # perform genre transfer with restored model

                # split batches
                # args.time_step # 'time step of pianoroll'
                # args.pitch_range # 'pitch range of pianoroll'
                # args.input_nc # '# of input image channels'
                # args.output_nc # '# of output image channels'


                # save midis
                origin = source_npy.reshape(1, source_npy.shape[0], source_npy.shape[1], 1)

                midi_path_res = os.path.join(self.res_dir,
                                             '{}_{}_{}.mid'.format(midi_name,
                                                                   args.checkpoint_name,
                                                                   args.which_direction))
                # midi_path_transfer = os.path.join(test_dir_mid, '{}_transfer.mid'.format(idx + 1))
                # midi_path_cycle = os.path.join(test_dir_mid, '{}_cycle.mid'.format(idx + 1))


                processed = []
                n_batches = origin.shape[1] // (self.time_step)
                print("n_batches", n_batches)

                # for i in range(n_batches)[:1]:
                #     print("processing batch:", i)
                #     if args.which_direction == 'AtoB':
                #         transfer = self.generator_A2B(origin[:, i * self.time_step : (i + 1) * self.time_step],
                #                                       training=False)
                #     else:
                #         transfer = self.generator_B2A(origin[:, i * self.time_step : (i + 1) * self.time_step],
                #                                       training=False)
                #     processed.append(transfer)

                # for i in range(n_batches)[:1]:
                #     processed.append(origin[:, i * self.time_step : (i + 1) * self.time_step])
                # processed = np.concatenate(processed, axis=0)
                processed = origin.reshape(-1,self.time_step, origin.shape[2], origin.shape[3])
                transfer = self.generator_A2B(processed, training=False)

                transfer = transfer.numpy()
                transfer = transfer.reshape(1, transfer.shape[0]* transfer.shape[1], transfer.shape[2], transfer.shape[3])

                # physical_devices = tf.config.list_physical_devices('GPU')
                # try:
                #     # Disable all GPUS
                #     tf.config.set_visible_devices([], 'GPU')
                #     visible_devices = tf.config.get_visible_devices()
                #     for device in visible_devices:
                #         assert device.device_type != 'GPU'
                # except:
                #     # Invalid device or cannot modify virtual devices once initialized.
                #     pass
                # transfer = self.generator_A2B(origin, training=False)
                # transfer = to_binary(transfer, 0.5)

                # transfer = origin
                # print("len(processed)", len(processed))
                # print(type(processed[0]), processed[0].shape)
                # transfer = np.concatenate(processed, axis=1)
                # print("transfer", type(transfer), transfer.shape, origin.shape, transfer.dtype, origin.dtype)
                # print("max val:", str(np.ndarray.max(transfer)))

                # print('midi_info', midi_info)
                # restore midi from results and save midi file to res directory
                transfer = to_binary(transfer, 0.5)

                # print("transfer", type(transfer), transfer.shape, origin.shape, transfer.dtype, origin.dtype)
                # print("max val:", str(np.ndarray.max(transfer)))

                # transfer = origin
                save_midis(transfer, midi_path_res, tempo=midi_info["tempo"], resolution=16)
                # save_midis(transfer, midi_path_res, tempo=midi_info["tempo"], beat_resolution=16)
                # save_midis_params(transfer, midi_path_res, tempo=midi_info["tempo"], resolution=16)
                print('Saved:',midi_path_res)
                return midi_path_res
            else:
                print('Error')
                return None

            # pass
        # pass
        print("All files converted correctly!")

    def train(self, args):

        # Data from domain A and B, and mixed dataset for partial and full models.
        dataA = glob('./datasets/{}/train/*.*'.format(self.dataset_A_dir))
        dataB = glob('./datasets/{}/train/*.*'.format(self.dataset_B_dir))
        data_mixed = None
        if self.model == 'partial':
            data_mixed = dataA + dataB
        if self.model == 'full':
            data_mixed = glob('./datasets/JCP_mixed/*.*')

        if args.continue_train:
            if self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint):
                print(" [*] Load checkpoint succeeded!")
            else:
                print(" [!] Load checkpoint failed...")

        counter = 1
        start_time = time.time()

        for epoch in range(args.epoch):

            # Shuffle training data
            np.random.shuffle(dataA)
            np.random.shuffle(dataB)
            if self.model != 'base' and data_mixed is not None:
                np.random.shuffle(data_mixed)

            # Get the proper number of batches
            batch_idxs = min(len(dataA), len(dataB)) // self.batch_size

            # learning rate starts to decay when reaching the threshold
            self.lr = self.lr if epoch < args.epoch_step else self.lr * (args.epoch-epoch) / (args.epoch-args.epoch_step)

            for idx in range(batch_idxs):

                # To feed real_data
                batch_files = list(zip(dataA[idx * self.batch_size:(idx + 1) * self.batch_size],
                                       dataB[idx * self.batch_size:(idx + 1) * self.batch_size]))
                batch_samples = [load_npy_data(batch_file) for batch_file in batch_files]
                batch_samples = np.array(batch_samples).astype(np.float32)  # batch_size * 64 * 84 * 2
                real_A, real_B = batch_samples[:, :, :, 0], batch_samples[:, :, :, 1]
                real_A = tf.expand_dims(real_A, -1)
                real_B = tf.expand_dims(real_B, -1)


                # generate gaussian noise for robustness improvement
                gaussian_noise = np.abs(np.random.normal(0,
                                                         self.sigma_d,
                                                         [self.batch_size,
                                                          self.time_step,
                                                          self.pitch_range,
                                                          self.input_c_dim])).astype(np.float32)

                if self.model == 'base':

                    with tf.GradientTape(persistent=True) as gen_tape, tf.GradientTape(persistent=True) as disc_tape:

                        fake_B = self.generator_A2B(real_A,
                                                    training=True)
                        cycle_A = self.generator_B2A(fake_B,
                                                     training=True)

                        fake_A = self.generator_B2A(real_B,
                                                    training=True)
                        cycle_B = self.generator_A2B(fake_A,
                                                     training=True)

                        [fake_A_sample, fake_B_sample] = self.pool([fake_A, fake_B])

                        DA_real = self.discriminator_A(real_A + gaussian_noise,
                                                       training=True)
                        DB_real = self.discriminator_B(real_B + gaussian_noise,
                                                       training=True)

                        DA_fake = self.discriminator_A(fake_A + gaussian_noise,
                                                       training=True)
                        DB_fake = self.discriminator_B(fake_B + gaussian_noise,
                                                       training=True)

                        DA_fake_sample = self.discriminator_A(fake_A_sample + gaussian_noise,
                                                              training=True)
                        DB_fake_sample = self.discriminator_B(fake_B_sample + gaussian_noise,
                                                              training=True)

                        # Generator loss
                        cycle_loss = self.L1_lambda * (abs_criterion(real_A, cycle_A) + abs_criterion(real_B, cycle_B))
                        g_A2B_loss = self.criterionGAN(DB_fake, tf.ones_like(DB_fake)) + cycle_loss
                        g_B2A_loss = self.criterionGAN(DA_fake, tf.ones_like(DA_fake)) + cycle_loss
                        g_loss = g_A2B_loss + g_B2A_loss - cycle_loss

                        # Discriminator loss
                        d_A_loss_real = self.criterionGAN(DA_real, tf.ones_like(DA_real))
                        d_A_loss_fake = self.criterionGAN(DA_fake_sample, tf.zeros_like(DA_fake_sample))
                        d_A_loss = (d_A_loss_real + d_A_loss_fake) / 2

                        d_B_loss_real = self.criterionGAN(DB_real, tf.ones_like(DB_real))
                        d_B_loss_fake = self.criterionGAN(DB_fake_sample, tf.zeros_like(DB_fake_sample))
                        d_B_loss = (d_B_loss_real + d_B_loss_fake) / 2
                        d_loss = d_A_loss + d_B_loss

                    # Calculate the gradients for generator and discriminator
                    generator_A2B_gradients = gen_tape.gradient(target=g_A2B_loss,
                                                                sources=self.generator_A2B.trainable_variables)
                    generator_B2A_gradients = gen_tape.gradient(target=g_B2A_loss,
                                                                sources=self.generator_B2A.trainable_variables)

                    discriminator_A_gradients = disc_tape.gradient(target=d_A_loss,
                                                                   sources=self.discriminator_A.trainable_variables)
                    discriminator_B_gradients = disc_tape.gradient(target=d_B_loss,
                                                                   sources=self.discriminator_B.trainable_variables)

                    # Apply the gradients to the optimizer
                    self.GA2B_optimizer.apply_gradients(zip(generator_A2B_gradients,
                                                            self.generator_A2B.trainable_variables))
                    self.GB2A_optimizer.apply_gradients(zip(generator_B2A_gradients,
                                                            self.generator_B2A.trainable_variables))

                    self.DA_optimizer.apply_gradients(zip(discriminator_A_gradients,
                                                          self.discriminator_A.trainable_variables))
                    self.DB_optimizer.apply_gradients(zip(discriminator_B_gradients,
                                                          self.discriminator_B.trainable_variables))

                    print('=================================================================')
                    print(("Epoch: [%2d] [%4d/%4d] time: %4.4f D_loss: %6.2f, G_loss: %6.2f, cycle_loss: %6.2f" %
                           (epoch, idx, batch_idxs, time.time() - start_time, d_loss, g_loss, cycle_loss)))

                else:

                    # To feed real_mixed
                    batch_files_mixed = data_mixed[idx * self.batch_size:(idx + 1) * self.batch_size]
                    batch_samples_mixed = [np.load(batch_file) * 1. for batch_file in batch_files_mixed]
                    real_mixed = np.array(batch_samples_mixed).astype(np.float32)

                    with tf.GradientTape(persistent=True) as gen_tape, tf.GradientTape(persistent=True) as disc_tape:
                        # print(real_A)
                        # print(real_A.shape)
                        fake_B = self.generator_A2B(real_A,
                                                    training=True)
                        cycle_A = self.generator_B2A(fake_B,
                                                     training=True)

                        fake_A = self.generator_B2A(real_B,
                                                    training=True)
                        cycle_B = self.generator_A2B(fake_A,
                                                     training=True)

                        [fake_A_sample, fake_B_sample] = self.pool([fake_A, fake_B])

                        DA_real = self.discriminator_A(real_A + gaussian_noise,
                                                       training=True)
                        DB_real = self.discriminator_B(real_B + gaussian_noise,
                                                       training=True)

                        DA_fake = self.discriminator_A(fake_A + gaussian_noise,
                                                       training=True)
                        DB_fake = self.discriminator_B(fake_B + gaussian_noise,
                                                       training=True)

                        DA_fake_sample = self.discriminator_A(fake_A_sample + gaussian_noise,
                                                              training=True)
                        DB_fake_sample = self.discriminator_B(fake_B_sample + gaussian_noise,
                                                              training=True)

                        DA_real_all = self.discriminator_A_all(real_mixed + gaussian_noise,
                                                               training=True)
                        DB_real_all = self.discriminator_B_all(real_mixed + gaussian_noise,
                                                               training=True)

                        DA_fake_sample_all = self.discriminator_A_all(fake_A_sample + gaussian_noise,
                                                                      training=True)
                        DB_fake_sample_all = self.discriminator_B_all(fake_B_sample + gaussian_noise,
                                                                      training=True)

                        # Generator loss
                        cycle_loss = self.L1_lambda * (abs_criterion(real_A, cycle_A) + abs_criterion(real_B, cycle_B))
                        g_A2B_loss = self.criterionGAN(DB_fake, tf.ones_like(DB_fake)) + cycle_loss
                        g_B2A_loss = self.criterionGAN(DA_fake, tf.ones_like(DA_fake)) + cycle_loss
                        g_loss = g_A2B_loss + g_B2A_loss - cycle_loss

                        # Discriminator loss
                        d_A_loss_real = self.criterionGAN(DA_real, tf.ones_like(DA_real))
                        d_A_loss_fake = self.criterionGAN(DA_fake_sample, tf.zeros_like(DA_fake_sample))
                        d_A_loss = (d_A_loss_real + d_A_loss_fake) / 2
                        d_B_loss_real = self.criterionGAN(DB_real, tf.ones_like(DB_real))
                        d_B_loss_fake = self.criterionGAN(DB_fake_sample, tf.zeros_like(DB_fake_sample))
                        d_B_loss = (d_B_loss_real + d_B_loss_fake) / 2
                        d_loss = d_A_loss + d_B_loss

                        d_A_all_loss_real = self.criterionGAN(DA_real_all, tf.ones_like(DA_real_all))
                        d_A_all_loss_fake = self.criterionGAN(DA_fake_sample_all, tf.zeros_like(DA_fake_sample_all))
                        d_A_all_loss = (d_A_all_loss_real + d_A_all_loss_fake) / 2
                        d_B_all_loss_real = self.criterionGAN(DB_real_all, tf.ones_like(DB_real_all))
                        d_B_all_loss_fake = self.criterionGAN(DB_fake_sample_all, tf.zeros_like(DB_fake_sample_all))
                        d_B_all_loss = (d_B_all_loss_real + d_B_all_loss_fake) / 2
                        d_all_loss = d_A_all_loss + d_B_all_loss
                        D_loss = d_loss + self.gamma * d_all_loss

                    # Calculate the gradients for generator and discriminator
                    generator_A2B_gradients = gen_tape.gradient(target=g_A2B_loss,
                                                                sources=self.generator_A2B.trainable_variables)
                    generator_B2A_gradients = gen_tape.gradient(target=g_B2A_loss,
                                                                sources=self.generator_B2A.trainable_variables)

                    discriminator_A_gradients = disc_tape.gradient(target=d_A_loss,
                                                                   sources=self.discriminator_A.trainable_variables)
                    discriminator_B_gradients = disc_tape.gradient(target=d_B_loss,
                                                                   sources=self.discriminator_B.trainable_variables)

                    discriminator_A_all_gradients = disc_tape.gradient(target=d_A_all_loss,
                                                                   sources=self.discriminator_A_all.trainable_variables)
                    discriminator_B_all_gradients = disc_tape.gradient(target=d_B_all_loss,
                                                                   sources=self.discriminator_B_all.trainable_variables)

                    # Apply the gradients to the optimizer
                    self.GA2B_optimizer.apply_gradients(zip(generator_A2B_gradients,
                                                            self.generator_A2B.trainable_variables))
                    self.GB2A_optimizer.apply_gradients(zip(generator_B2A_gradients,
                                                            self.generator_B2A.trainable_variables))

                    self.DA_optimizer.apply_gradients(zip(discriminator_A_gradients,
                                                          self.discriminator_A.trainable_variables))
                    self.DB_optimizer.apply_gradients(zip(discriminator_B_gradients,
                                                          self.discriminator_B.trainable_variables))

                    self.DA_all_optimizer.apply_gradients(zip(discriminator_A_all_gradients,
                                                              self.discriminator_A_all.trainable_variables))
                    self.DB_all_optimizer.apply_gradients(zip(discriminator_B_all_gradients,
                                                              self.discriminator_B_all.trainable_variables))

                    print('=================================================================')
                    print(("Epoch: [%2d] [%4d/%4d] time: %4.4f D_loss: %6.2f, G_loss: %6.2f" %
                           (epoch, idx, batch_idxs, time.time() - start_time, D_loss, g_loss)))

                counter += 1

                # generate samples during training to track the learning process
                if np.mod(counter, args.print_freq) == 1:
                    sample_dir = os.path.join(self.sample_dir,
                                              '{}2{}_{}_{}_{}'.format(self.dataset_A_dir,
                                                                      self.dataset_B_dir,
                                                                      self.now_datetime,
                                                                      self.model,
                                                                      self.sigma_d))
                    if not os.path.exists(sample_dir):
                        os.makedirs(sample_dir)

                    # to binary, 0 denotes note off, 1 denotes note on
                    samples = [to_binary(real_A, 0.5),
                               to_binary(fake_B, 0.5),
                               to_binary(cycle_A, 0.5),
                               to_binary(real_B, 0.5),
                               to_binary(fake_A, 0.5),
                               to_binary(cycle_B, 0.5)]

                    self.sample_model(samples=samples,
                                      sample_dir=sample_dir,
                                      epoch=epoch,
                                      idx=idx)

                if np.mod(counter, args.save_freq) == 1:
                    self.checkpoint_manager.save(counter)

    def sample_model(self, samples, sample_dir, epoch, idx):

        print('generating samples during learning......')

        if not os.path.exists(os.path.join(sample_dir, 'B2A')):
            os.makedirs(os.path.join(sample_dir, 'B2A'))
        if not os.path.exists(os.path.join(sample_dir, 'A2B')):
            os.makedirs(os.path.join(sample_dir, 'A2B'))

        save_midis(samples[0], './{}/A2B/{:02d}_{:04d}_origin.mid'.format(sample_dir, epoch, idx))
        save_midis(samples[1], './{}/A2B/{:02d}_{:04d}_transfer.mid'.format(sample_dir, epoch, idx))
        save_midis(samples[2], './{}/A2B/{:02d}_{:04d}_cycle.mid'.format(sample_dir, epoch, idx))
        save_midis(samples[3], './{}/B2A/{:02d}_{:04d}_origin.mid'.format(sample_dir, epoch, idx))
        save_midis(samples[4], './{}/B2A/{:02d}_{:04d}_transfer.mid'.format(sample_dir, epoch, idx))
        save_midis(samples[5], './{}/B2A/{:02d}_{:04d}_cycle.mid'.format(sample_dir, epoch, idx))

    def test(self, args):

        if args.which_direction == 'AtoB':
            sample_files = glob('./datasets/{}/test/*.*'.format(self.dataset_A_dir))
        elif args.which_direction == 'BtoA':
            sample_files = glob('./datasets/{}/test/*.*'.format(self.dataset_B_dir))
        else:
            raise Exception('--which_direction must be AtoB or BtoA')
        sample_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[-1]))

        if self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint):
            print(" [*] Load checkpoint succeeded!")
        else:
            print(" [!] Load checkpoint failed...")

        test_dir_mid = os.path.join(args.test_dir, '{}2{}_{}_{}_{}/{}/mid'.format(self.dataset_A_dir,
                                                                                  self.dataset_B_dir,
                                                                                  self.now_datetime,
                                                                                  self.model,
                                                                                  self.sigma_d,
                                                                                  args.which_direction))
        if not os.path.exists(test_dir_mid):
            os.makedirs(test_dir_mid)

        test_dir_npy = os.path.join(args.test_dir, '{}2{}_{}_{}_{}/{}/npy'.format(self.dataset_A_dir,
                                                                                  self.dataset_B_dir,
                                                                                  self.now_datetime,
                                                                                  self.model,
                                                                                  self.sigma_d,
                                                                                  args.which_direction))
        if not os.path.exists(test_dir_npy):
            os.makedirs(test_dir_npy)

        for idx in range(len(sample_files)):
            print('Processing midi: ', sample_files[idx])
            sample_npy = np.load(sample_files[idx]) * 1.

            # save midis
            origin = sample_npy.reshape(1, sample_npy.shape[0], sample_npy.shape[1], 1)
            midi_path_origin = os.path.join(test_dir_mid, '{}_origin.mid'.format(idx + 1))
            midi_path_transfer = os.path.join(test_dir_mid, '{}_transfer.mid'.format(idx + 1))
            midi_path_cycle = os.path.join(test_dir_mid, '{}_cycle.mid'.format(idx + 1))
            print(origin.shape, type(origin))
            break
            if args.which_direction == 'AtoB':

                transfer = self.generator_A2B(origin,
                                              training=False)
                cycle = self.generator_B2A(transfer,
                                           training=False)

            else:

                transfer = self.generator_B2A(origin,
                                              training=False)
                cycle = self.generator_A2B(transfer,
                                           training=False)

            save_midis(origin, midi_path_origin)
            save_midis(transfer, midi_path_transfer)
            save_midis(cycle, midi_path_cycle)

            # save npy files
            npy_path_origin = os.path.join(test_dir_npy, 'origin')
            npy_path_transfer = os.path.join(test_dir_npy, 'transfer')
            npy_path_cycle = os.path.join(test_dir_npy, 'cycle')

            if not os.path.exists(npy_path_origin):
                os.makedirs(npy_path_origin)
            if not os.path.exists(npy_path_transfer):
                os.makedirs(npy_path_transfer)
            if not os.path.exists(npy_path_cycle):
                os.makedirs(npy_path_cycle)

            np.save(os.path.join(npy_path_origin, '{}_origin.npy'.format(idx + 1)), origin)
            np.save(os.path.join(npy_path_transfer, '{}_transfer.npy'.format(idx + 1)), transfer)
            np.save(os.path.join(npy_path_cycle, '{}_cycle.npy'.format(idx + 1)), cycle)

    def test_famous(self, args):

        song = np.load('./datasets/famous_songs/P2C/merged_npy/YMCA.npy')

        if self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint):
            print(" [*] Load checkpoint succeeded!")
        else:
            print(" [!] Load checkpoint failed...")

        if args.which_direction == 'AtoB':
            transfer = self.generator_A2B(song,
                                          training=False)
        else:
            transfer = self.generator_B2A(song,
                                          training=False)

        save_midis(transfer, './datasets/famous_songs/P2C/transfer/YMCA.mid', 127)
        np.save('./datasets/famous_songs/P2C/transfer/YMCA.npy', transfer)