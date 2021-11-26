import torch
from torch.utils.data import Dataset
import leaf_audio_pytorch.frontend as frontend
#from segmentation import segment_cough
import librosa  

import pandas as pd
import os
import pydub
import numpy as np
import logging
from scipy.stats import kurtosis, entropy
from coughvid.audio_processing import (
        normalize_audio, extract_frames, extract_other_features, generate_feature_matrix)

logger = logging.getLogger(__name__)


class CoughvidDataset(Dataset):
    def __init__(self,
                 data_dir,
                 csv_file='metadata_compiled.csv', 
                 mask_loc=None,
                 filter_data=True, 
                 get_features=True,
                 sample_rate=48000, 
                 frame_length=1024, 
                 frames=50,
                 samples_per_class=None):

        # load dataframe
        assert os.path.isdir(data_dir), f'Data directory {data_dir} does not exist.'
        self.data_dir = data_dir
        csv_path = os.path.join(data_dir, csv_file)
        assert os.path.isfile(csv_path), f'CSV file {csv_path} does not exist.'
        with open(csv_path, 'r') as f:
            self.dataframe = pd.read_csv(f)
        self.__convert_to_numeric__(self.dataframe)
        self.audio_extensions = ['.webm', '.ogg']
        self.labels = ['status']#, 'SNR', 'status', 'age']# , 'respiratory_condition', 'gender']
        
        #load mask arrays
        self.mask_loc = mask_loc if mask_loc else data_dir
        #assert os.path.isfile(self.mask_loc), f'Mask file {self.mask_loc} does not exist. Calculating masks on the fly.'
        self.mask_array = np.load(mask_loc, allow_pickle = True) if os.path.isfile(self.mask_loc) else None
        if not self.mask_array: print(f'Mask file {self.mask_loc} does not exist. Calculating masks on the fly.')


        # get only records that have a COVID status label and a cough-detected above 0.8. Loading all the files takes too long
        assert filter_data, f'WARNING: All {len(self)} records have been selected for loading.'
        if filter_data:
            status_groups = [0,1]
            status = np.isin(self.dataframe['status'],status_groups)#['healthy','symptomatic','COVID-19'])
            cough_detected = self.dataframe['cough_detected'] > 0.8 # recommended threshold from https://www.nature.com/articles/s41597-021-00937-4

            self.dataframe = self.dataframe[ np.logical_and(status,cough_detected) ]

            # obtain at least samples_per_class per class
            if samples_per_class:
                samples = [self.dataframe[self.dataframe['status'] == i].head(samples_per_class) for i in status_groups]
                self.dataframe = pd.concat(samples)

            print(f'{len(self)} records ready to load across {len(status_groups)} groups.')

        # set frame parameters and MFCC module
        self.frame_length = frame_length
        self.frames       = frames
        self.sample_rate  = sample_rate
        self.get_features = get_features

    def __getitem__(self, index):
        entry = self.dataframe.iloc[index]
        uuid = entry['uuid']
        audio = None
        for ext in self.audio_extensions:
            filename = os.path.join(self.data_dir, f'{uuid}{ext}')
            #logger.debug(filename)
            if os.path.isfile(filename):
                audio = pydub.AudioSegment.from_file(filename)
                break
        assert audio is not None, f"No audio found with uuid {uuid}"
        audio = np.array(audio.get_array_of_samples(), dtype='int64')
        labels = torch.IntTensor(entry[self.labels])[0]

        # return raw audio and labels unless self.get_features
        if not self.get_features: return audio, labels

        # first, normalize audio
        audio = normalize_audio(audio)

        # second, load segmented array and apply mask
        mask = 1#self.mask_array[index] if self.mask_array else segment_cough(audio,self.sample_rate)[1]
        masked_audio = np.ma.masked_array(audio,1-mask) # 0 is uncensored, 1 is censored

        # drop samples too short to analyze
        if len(masked_audio.compressed()) < self.frame_length:
            print (f'Skipping sample {uuid} as it only contains {len(masked_audio.compressed())} frames.')
            if index < len(self):
                return self.__getitem__(index+1)
            else:
                return self.__getitem__(index-1)

        # extract self.frames number of frames of self.frame_length length from masked audio
        frames = extract_frames(masked_audio.compressed())#,uuid)

        #mels = [self.mfcc(torch.from_numpy(frame).type(torch.FloatTensor)).flatten().tolist()[:38] for frame in frames]
        #mel_d= self.mfcc_delta(np.array(mels),2)
        #mel_dd = self.mfcc_delta(mel_d,2)

        # compute features
        mfcc        = librosa.feature.mfcc(frames.flatten(), sr=self.sample_rate, n_mfcc=26, n_mels=40, n_fft=512, hop_length=self.frame_length, power=2,center=False)
        #mfcc       -= np.mean(mfcc)
        mfcc_delta  = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        other_feat  = np.array([extract_other_features(frame) for frame in frames]).T

        features = np.concatenate((mfcc, mfcc_delta, mfcc_delta2, other_feat)) # shape should be (n_mfcc * 3 + 3, self.frames)

        # normalization steps

        # resize to interval [0,1] along time axis (didn't work)
        #features = (features - np.min(features,axis=1)[None,...].T) / (np.max(features,axis=1)[None,...].T-np.min(features,axis=1)[None,...].T)
        #features = (features - np.min(features)) / (np.max(features)-np.min(features))

        # return standard deviations
        #features = (features - np.mean(features)) / np.std(features)

        # return as image with values from [0,255]
        #features = self.spec_to_image(features)

        return features, labels

    def __len__(self):
        return self.dataframe.shape[0]

    def __convert_to_numeric__(self, dataframe):
        #respiratory_condition_map = {'False': 0, 'True': 1, 'NaN': -1}
        status_map = {'healthy': 0, 'symptomatic': 2, "COVID-19": 1, 'NaN': -1}
        #gender_map = {'female': 0, 'male': 1, 'other': 2, 'NaN': -1}
        for key, value in status_map.items():
            dataframe.loc[dataframe['status'] == key, 'status'] = value
