import pandas as pd
import os


def delete_empty(df, audiodir):
    ids = list(df['id'])
    for _id in ids:
        audiofile = os.path.join(audiodir, f'{_id}.wav')
        if not os.path.isfile(audiofile):
            print(f"Coudn't find file {audiofile}")
        else:
            size = os.path.getsize(audiofile)
            if size < 100:
                print(f"Deleting file {audiofile} with size {size}")
                os.remove(audiofile)
                print(f"{audiofile} deleted")
                df = df[df['id'] != _id]
    return df


if __name__ == "__main__":
    csvfile = 'extracted_data.csv'
    df = pd.read_csv(csvfile)
    audiodir = 'audio'
    nonempty_df = delete_empty(df, audiodir)
    outfile = 'filtered_data.csv'
    nonempty_df.to_csv(outfile)
    print(f"Wrote new csv to {outfile}")
