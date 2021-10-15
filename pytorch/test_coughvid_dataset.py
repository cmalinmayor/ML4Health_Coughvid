import time
from .coughvid_dataset import CoughvidDataset
from torch.utils.data import DataLoader

PATH_TO_DATASET = "D:\COUGHVID_public_dataset\public_dataset"


class TestCoughvidDataset:

    def test_simple(self):
        dataset = CoughvidDataset(PATH_TO_DATASET, 'metadata_compiled.csv')
        print(len(dataset))
        entry, audio = dataset[0]
        print(entry)
        print(audio.shape)

    def test_data_loader(self):
        num_batches = 1000
        batch_size = 1
        num_workers = 1
        dataset = CoughvidDataset(PATH_TO_DATASET, 'metadata_compiled.csv')
        dataloader = DataLoader(dataset, num_workers=num_workers, batch_size=batch_size)
        time_start = time.time()
        current_batch = 0
        for audio, labels in dataloader:
            print(f'Batch {current_batch} has audio shape {audio.shape} and label shape {labels.shape}')
            current_batch += 1
            if current_batch > num_batches:
                break
        time_end = time.time()
        total_time = time_end - time_start
        print(f'Total time (sec): {total_time} ({total_time/num_batches} per batch,'
              f' {total_time/num_batches/batch_size} per sample with {num_workers} workers)')

        ''' this can load 1000 samples in 32 seconds on Caroline's machine with 10 workers
         -> about 2k per minute
         -> about 13 minutes to iterate over all samples
         probably a little less because there is about 10 seconds of overhead, and then only 22 seconds for iterating
         I think a lot of the speed up compared to what Filip was doing comes from the parallelization - 
         with only 1 worker it takes 120 seconds to load 1000 samples (which would be almost an hour for 27k)
        '''



