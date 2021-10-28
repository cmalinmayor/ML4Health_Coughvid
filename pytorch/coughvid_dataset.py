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
        self.sample_rate = sample_rate

        self.mfcc = MFCC(sample_rate=self.sample_rate, n_mfcc=20, melkwargs={"n_fft": 2048, "hop_length": 512, "power": 2})
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



        return audio, labels

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
        pass

    def mfcc_acceleration(self,mfccs):
        pass

    ### FUNCTIONS FOR FRAME EXTRACTION ###
    def frame_skip(self,audio):
        return np.ceil(len(audio)*1.0/self.frames)

    def frame_extract(self,audio,mask):
        pass

