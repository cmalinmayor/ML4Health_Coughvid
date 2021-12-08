import torch
from torch.utils.data import Dataset

import pandas as pd
import os
import numpy as np
import logging

logger = logging.getLogger(__name__)


class CoswaraEmbeddingDataset(Dataset):
    def __init__(self,
                 data_dir,
                 csv_file='filtered_data.csv',
                 sample_rate=16000,
                 feature_d=512,
                 unit_sec=6.0):
        # define internal variables
        self.labels = ['covid_status']
        self.sample_rate = sample_rate
        self.feature_d = feature_d
        self.unit_sec = unit_sec

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
        data = None
        filename = os.path.join(self.data_dir, 'embeddings', f'e_{self.feature_d}_{self.unit_sec}_{uuid}.wav.npy')
        if os.path.isfile(filename):
            data = np.load(filename)
        else:
            assert data is not None, f"No audio found with uuid {uuid}"
            print(data.dtype)
        labels = torch.IntTensor(entry[self.labels])[0]

        return data.astype(np.float32), labels

    def __len__(self):
        return self.dataframe.shape[0]

    def __convert_to_numeric__(self, dataframe):
        status_map = {'healthy': 0, 'positive_moderate': 1, "positive_mild": 1}
        for key, value in status_map.items():
            dataframe.loc[dataframe['covid_status'] == key,
                          'covid_status'] = value
