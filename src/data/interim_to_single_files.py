import pandas as pd
from pathlib import Path

def split_local_data_to_single_files(data_dir, extension='.rna'):
    target_file_paths = Path(data_dir, dataset).glob('*.local')
    for target_file_path in target_file_paths:
        [Path(data_dir, f"{index+1}{extension}").resolve().write_text(target.rstrip('\n')) for index, target in enumerate(target_file_path.read_text().rstrip('\n').split('\n'))]

def split_via_dataframe(data_dir, dataset, in_extension='.interim', out_extension='.rna'):
    path = Path(data_dir, dataset, f"{str(dataset).split('_')[-1]}{in_extension}").resolve()
    print('Read interim data')
    df = pd.read_csv(path, sep='\t')
    print('Write single files')
    # [pd.DataFrame([(current_index +1, row[0], row[1], row[2], row[3], row[4], row[5])]).to_csv(path_or_buf=Path(data_dir, dataset, f"{current_index+1}{out_extension}"), sep='\t', index=False) for current_index, row in enumerate(zip(df['structure'], df['sequence'], df['local_random'], df['local_motif'], df['gc_content'], df['mfe']))]
    [Path(data_dir, dataset, f"{index+1}{out_extension}").resolve().write_text(f"{index+1}\t{row[0]}\t{row[1]}\t{row[2]}\t{row[3]}\t{row[4]}\t{row[5]}") for index, row in enumerate(zip(df['structure'], df['sequence'], df['local_random'], df['local_motif'], df['gc_content'], df['mfe']))]




if __name__ == '__main__':
    # split_local_data_to_single_files('data/rfam_learn_local/test')
    # split_local_data_to_single_files('data/rfam_learn_local/train')
    # split_local_data_to_single_files('data/rfam_learn_local/validation')
    # print("split test data")
    split_via_dataframe('data/', 'rfam_learn_local_min_100_max_500_test')
    # print('split training data')
    split_via_dataframe('data/', 'rfam_learn_local_min_500_test')
    # print('split validation data')
    # split_via_dataframe('data/rfam_learn_local', 'validation')
