import torch
from torch.utils.data import Dataset
from torchaudio.transforms import MFCC
import leaf_audio_pytorch.frontend as frontend

import pandas
import os
import pydub
import numpy as np
import logging
from scipy.stats import kurtosis, entropy

logger = logging.getLogger(__name__)


class CoughvidDataset(Dataset):
    def __init__(self,
                 data_dir,
                 csv_file='metadata_compiled.csv', 
                 filter_data=True, 
                 get_features=True,
                 sample_rate=48000, 
                 frame_length=1024, 
                 frames=100):

        # load dataframe
        assert os.path.isdir(data_dir), f'Data directory {data_dir} does not exist.'
        self.data_dir = data_dir
        csv_path = os.path.join(data_dir, csv_file)
        assert os.path.isfile(csv_path), f'CSV file {csv_path} does not exist.'
        with open(csv_path, 'r') as f:
            self.dataframe = pandas.read_csv(f)
        self.__convert_to_numeric__(self.dataframe)
        self.audio_extensions = ['.webm', '.ogg']
        self.labels = ['cough_detected']#, 'SNR', 'status', 'age']# , 'respiratory_condition', 'gender']


        # get only records that have a COVID status label and a cough-detected above 0.8. Loading all the files takes too long
        assert filter_data, f'WARNING: All {len(self)} records have been selected for loading.'
        if filter_data:
            status = np.isin(self.dataframe['status'],[0,1,2])#['healthy','symptomatic','COVID-19'])
            cough_detected = self.dataframe['cough_detected'] > 0.8 # recommended threshold from https://www.nature.com/articles/s41597-021-00937-4

            self.dataframe = self.dataframe[ np.logical_and(status,cough_detected) ]

            print(f'{len(self)} records ready to load.')

        # set frame parameters and MFCC module
        self.frame_length = frame_length
        self.frames       = frames
        self.sample_rate  = sample_rate
        self.get_features = get_features

        self.mfcc = MFCC(sample_rate=self.sample_rate, n_mfcc=20, melkwargs={'n_mels':39,'center':False,"n_fft": 400, "power": 2})
        #torch_mfcc = mfcc_module(torch.tensor(audio))

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
        labels = np.array(entry[self.labels], dtype=np.float32)

        # return raw audio and labels unless self.get_features
        if not self.get_features: return audio, labels

        audio = self.normalize_audio(audio)
        # segmented array
        mask = 1 # placeholder
        masked_audio = np.ma.masked_array(audio,1-mask) # 0 is uncensored, 1 is censored

        frames = self.extract_frames(masked_audio)


        features = [self.extract_features(frame) for frame in frames]

        return features, labels



    def __len__(self):
        return self.dataframe.shape[0]

    def __convert_to_numeric__(self, dataframe):
        #respiratory_condition_map = {'False': 0, 'True': 1, 'NaN': -1}
        status_map = {'healthy': 0, 'symptomatic': 1, "COVID-19": 2, 'NaN': -1}
        #gender_map = {'female': 0, 'male': 1, 'other': 2, 'NaN': -1}
        for key, value in status_map.items():
            dataframe.loc[dataframe['status'] == key, 'status'] = value

    ### FUNCTIONS FOR FEATURE EXTRACTION ###
    def normalize_audio(self,audio):
        '''Normalize the audio signal according to the formula in
        https://www.sciencedirect.com/science/article/pii/S0010482521003668?via%3Dihub
        '''
        return 0.9 * audio / np.max(audio)

    def zcr(self,frame):
        '''Calculate the number of times the signal passes through zero,
        a.k.a. the zero-crossing rate.
        '''
        zero_crosses = np.nonzero(np.diff(frame > 0))[0]
        return zero_crosses

    def log_energy(self,frame):
        '''Calculate the log energy of the audio signal.
        '''
        return np.log2(np.sum(np.power(frame,2)))

    def mfcc_velocity(self,mfccs):
        return -1

    def mfcc_acceleration(self,mfccs):
        return -1

    def extract_frames(self,masked_audio):
        '''Extract self.frames number of frames of self.frame_length length.
        '''
        frame_skip = np.ceil(len(masked_audio)*1.0/self.frames)
        valid_samples = masked_audio.compressed()

        return [valid_samples[int(i):int(i+self.frame_length)] for i in np.arange(0,len(valid_samples),frame_skip)]

    def extract_features(self,frame):
        '''Extract mel-cepstral coefficients, their acceleration, their velocity,
        frame kurtosis, frame log energy and frame zero-crossing rate.
        '''
        frame = np.array(frame)
        #assert frame.shape == (self.frame_length,1), f'Unexpected shape: {frame.shape}'
        tframe = torch.from_numpy(frame)
        tframe = tframe.type(torch.FloatTensor)
        mfccs = self.mfcc(tframe)
        return [mfccs, self.mfcc_velocity(mfccs), self.mfcc_acceleration(mfccs), kurtosis(frame), self.log_energy(frame), self.zcr(frame)]
