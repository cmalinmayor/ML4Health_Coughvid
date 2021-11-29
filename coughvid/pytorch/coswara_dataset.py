import torch
from torch.utils.data import Dataset
from torchaudio.transforms import MFCC
import librosa

import pandas as pd
import os
import pydub
import numpy as np
import logging
from coughvid.audio_processing import (
        normalize_audio, extract_frames, extract_other_features, generate_feature_matrix)

from data_augmentation.data_augmentation import DataAugmentation

logger = logging.getLogger(__name__)

class CoswaraDataset(Dataset):
    def __init__(self,
                 data_dir,
                 csv_file='filtered_data.csv',
                 get_features=True,
                 filter_data=True, 
                 sample_rate=44100,
                 frame_length=1024,
                 frames=50,
                 samples_per_class=None):
        # define internal variables
        self.labels = ['covid_status']
        self.frame_length = frame_length
        self.frames = frames
        self.sample_rate = sample_rate
        self.get_features = get_features
        self.samples_per_class = samples_per_class
        self.n_fft = 512

        # load dataframe
        assert os.path.isdir(data_dir),\
            f'Data directory {data_dir} does not exist.'
        self.data_dir = data_dir
        csv_path = os.path.join(data_dir, csv_file)
        assert os.path.isfile(csv_path), f'CSV file {csv_path} does not exist.'
        with open(csv_path, 'r') as f:
            self.dataframe = pd.read_csv(f)

        # generate labels for training
        self.__convert_to_numeric__(self.dataframe)

    def __getitem__(self, index):
        entry = self.dataframe.iloc[index]
        uuid = entry['id']
        audio = None
        filename = os.path.join(self.data_dir, 'audio', f'{uuid}.wav')
        if os.path.isfile(filename):
            audio = pydub.AudioSegment.from_file(filename)
        else:
            assert audio is not None, f"No audio found with uuid {uuid}"
        audio = np.array(audio.get_array_of_samples(), dtype=np.int64)
        labels = torch.IntTensor(entry[self.labels])[0]

        # return raw audio and labels unless self.get_features
        if not self.get_features:
            return audio, labels

        # first, normalize audio
        audio = normalize_audio(audio)

        # drop samples too short to analyze
        if len(audio) < self.frame_length:
            print(f'Skipping sample {uuid} as it only contains '
                  f'{len(audio)} elements.')
            if index < len(self):
                return self.__getitem__(index+1)
            else:
                return None

        # apply data augmentation methods
        da = DataAugmentation(audio)
        # randomly decide method
        decision = np.random.randint(2, size=3)
        noise, lp, hp = decision[0], decision[1], decision[2]

        if noise:
            audio = da.apply_gaussian_noise()
        if lp:
            audio = da.apply_lp()
        if hp:
            audio = da.apply_hp()

        # extract self.frames number of frames of self.frame_length length
        frames = extract_frames(audio, self.frames, self.frame_length)

        # compute features
        mfcc = librosa.feature.mfcc(
                frames.flatten(), 
                sr=self.sample_rate,
                n_mfcc=26, 
                n_mels=40, 
                n_fft=512,
                hop_length=self.frame_length,
                power=2, 
                center=False,
                fmax=8192)
        mfcc_delta  = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        other_feat  = np.array([extract_other_features(frame)
                               for frame in frames]).T

        features = np.concatenate((mfcc, mfcc_delta, mfcc_delta2, other_feat))
        # shape should be (n_mfcc * 3 + 3, self.frames)

        return features.astype(np.float32), labels

    def __len__(self):
        return self.dataframe.shape[0]

    def __convert_to_numeric__(self, dataframe):
        status_map = {'healthy': 0, 'positive_moderate': 1, "positive_mild": 1}
        for key, value in status_map.items():
            dataframe.loc[dataframe['covid_status'] == key,
                          'covid_status'] = value
