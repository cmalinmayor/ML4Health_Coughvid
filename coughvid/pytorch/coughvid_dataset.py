import torch
from torch.utils.data import Dataset
from torchaudio.transforms import MFCC
import leaf_audio_pytorch.frontend as frontend
from segmentation import segment_cough
import librosa  

import pandas as pd
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
            status_groups = [0,2]
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


        n_fft = 512
        #frame_length = n_fft / self.sample_rate * 1000.0
        #frame_shift = frame_length / 2.0
        self.mfcc = MFCC(sample_rate=self.sample_rate, n_mfcc=39, melkwargs={'center':True, "power": 2,'n_fft':n_fft,'n_mels':40}) #'n_mels':39,"n_fft": 200,
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
        labels = torch.IntTensor(entry[self.labels])[0]

        # return raw audio and labels unless self.get_features
        if not self.get_features: return audio, labels

        # first, normalize audio
        audio = self.normalize_audio(audio)

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
        frames = self.extract_frames(masked_audio.compressed())#,uuid)

        #mels = [self.mfcc(torch.from_numpy(frame).type(torch.FloatTensor)).flatten().tolist()[:38] for frame in frames]
        #mel_d= self.mfcc_delta(np.array(mels),2)
        #mel_dd = self.mfcc_delta(mel_d,2)

        # compute features
        mfcc        = librosa.feature.mfcc(frames.flatten(), sr=self.sample_rate, n_mfcc=26, n_mels=40, n_fft=512, hop_length=self.frame_length, power=2,center=False)
        #mfcc       -= np.mean(mfcc)
        mfcc_delta  = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        other_feat  = np.array([self.extract_other_features(frame) for frame in frames]).T

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
        status_map = {'healthy': 0, 'symptomatic': 1, "COVID-19": 2, 'NaN': -1}
        #gender_map = {'female': 0, 'male': 1, 'other': 2, 'NaN': -1}
        for key, value in status_map.items():
            dataframe.loc[dataframe['status'] == key, 'status'] = value

    ### FUNCTIONS FOR FEATURE EXTRACTION ###
    def spec_to_image(self, spec, eps=1e-6):
        mean = np.mean(spec)
        std = np.std(spec)
        spec_norm = (spec - mean) / (std + eps)
        spec_min, spec_max = np.min(spec), np.max(spec)
        spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
        spec_scaled = spec_scaled.astype(np.uint8)
        return spec_scaled

    def normalize_audio(self,audio):
        '''Normalize the audio signal according to the formula in
        https://www.sciencedirect.com/science/article/pii/S0010482521003668?via%3Dihub
        '''
        return 0.9 * audio / np.max(audio)

    def zcr(self,frame):
        '''Calculate the number of times the signal passes through zero,
        a.k.a. the zero-crossing rate.
        '''
        zero_crosses = len(np.nonzero(np.diff(frame > 0))[0])
        return zero_crosses

    def log_energy(self,frame):
        '''Calculate the log energy of the audio signal.
        '''
        return np.log2(max(1,np.sum(np.power(frame,2))))

    def extract_frames(self,valid_samples,uuid=None):
        '''Extract self.frames number of frames of self.frame_length length.
        '''
        #if not len(valid_samples) >= self.frame_length * self.frames: print( f'WARNING: {len(valid_samples)} frames found, need at least {self.frame_length * self.frames}' )

        #frame_skip = int(np.ceil(len(valid_samples)*1.0/self.frames))
        #assert frame_skip > 0

        frames = []

        for i in np.linspace(0,len(valid_samples)-1,self.frames,endpoint=False):
            frame = valid_samples[int(i):int(i)+self.frame_length]


            if len(frame) < self.frame_length:
                print(f'WARNING: Unexpected frame length encountered at {int(i)} of {len(valid_samples)}: {len(frame)}. Padding {self.frame_length-len(frame)} frames.')

                frame = np.pad(frame, (0, max(0,self.frame_length-len(frame))), 'constant')
            elif len(frame) > self.frame_length: 
                frame = frame[:self.frame_length-1]

            #assert len(frame) == self.frame_length, f'Frame length of {len(frame)} detected in {uuid}.'

            frames += [frame]

        #assert len(frames) == self.frames, f'Only {len(frames)} frames extracted, need {self.frames}.'

        return np.array(frames,dtype=np.float32)

    def extract_other_features(self,frame):
        '''Extract frame kurtosis, frame log energy and frame zero-crossing rate.
        '''
        features =  np.array([kurtosis(frame), self.log_energy(frame), self.zcr(frame)])
        return features
