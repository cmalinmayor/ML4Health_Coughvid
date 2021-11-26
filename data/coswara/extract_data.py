import os
import subprocess
import glob
import pandas as pd
import logging
import shutil
from tqdm import tqdm

logger = logging.getLogger(__name__)

'''
This script creates a dataset of deep coughs where each filename is the id.
The dataset only includes samples with "covid_status" equal to "healthy", "positive_mild", or "positive_moderate".
'''


def filter_by_covid_status(df):
    include = ['healthy', 'positive_moderate', 'positive_mild']
    df = df[df['covid_status'].isin(include)]
    return df


def get_metadata(data_download, extracted_data_dir):
    fname = 'combined_data.csv'
    orig_path = os.path.join(data_download, fname)
    new_fname = 'extracted_data.csv'
    new_path = os.path.join(extracted_data_dir, new_fname)

    df = pd.read_csv(orig_path)
    df = filter_by_covid_status(df)
    df.to_csv(new_path)
    return df


def get_data_from_dir(d, ids, extracted_data_dir, coswara_data_dir):
    temp_dir = os.path.join(extracted_data_dir, "tmp")
    os.makedirs(temp_dir, exist_ok=True)
    p = subprocess.Popen('cat {}/{}/*.tar.gz.* |tar -xvz -C {}/'.format(coswara_data_dir, d, temp_dir), shell=True)
    p.wait()

    root_dir = os.path.join(temp_dir, os.listdir(temp_dir)[0])

    for extracted_sample in os.listdir(root_dir):
        if extracted_sample in ids:
            orig_cough = os.path.join(root_dir, extracted_sample, 'cough-heavy.wav')
            print(f'found one! {orig_cough}')
            if os.path.isfile(orig_cough):
                print("file exists")
                new_cough = os.path.join(extracted_data_dir, f'{extracted_sample}.wav')
                shutil.move(orig_cough, new_cough)
                print(f'moved cough to {new_cough}')
    shutil.rmtree(temp_dir)


if __name__ == '__main__':
    # Local Path of iiscleap/Coswara-Data Repo
    data_download = os.path.abspath('../coswara_orig')
    extracted_data_dir = os.path.abspath('.')

    if not os.path.exists(data_download):
        raise("Check the Coswara dataset directory!")
    if not os.path.exists(extracted_data_dir):
        os.makedirs(extracted_data_dir)

    dirs_to_extract = set(map(os.path.basename,
                              glob.glob('{}/202*'.format(data_download))))

    metadata = get_metadata(data_download, extracted_data_dir)
    path_to_audio = os.path.join(extracted_data_dir, 'audio')
    for d in tqdm(dirs_to_extract):
        get_data_from_dir(d, set(metadata['id']), path_to_audio, data_download)
    print("Extraction process complete!")
