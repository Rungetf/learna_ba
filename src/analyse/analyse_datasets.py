from pathlib import Path
from collections import Counter
import pandas as pd

datasets = ['rfam_local_test', 'rfam_local_validation', 'rfam_local_train', 'rfam_learn_local_min_500_test', 'rfam_learn_local_min_100_max_500_test', 'rfam_local_long_train', 'rfam_local_short_train', 'rfam_local_min_400_max_1000_test', 'rfam_local_min_1000_test']

def get_dataset(data_path, dataset):
    return Path(data_path, dataset).glob('*.rna')

def analyse_datasets(data_path):
    length_distributions = {}
    dataset_paths = [get_dataset(data_path, dataset) for dataset in datasets]

    for dataset in dataset_paths:
        counter = 0
        length = []
        for path in dataset:
            if counter == 0:
                print(f"process dataset {path.parent.stem}")
                counter += 1
            length.append(len(path.read_text().split('\t')[1]))
        dataset_length_distribution = Counter(length)
        print(f"Minimum length in {path.parent.stem}: {min([entry for entry in dataset_length_distribution])}")
        print(f"Maximum length in {path.parent.stem}: {max([entry for entry in dataset_length_distribution])}")
        length_distributions[path.parent.stem] = dataset_length_distribution
    df = pd.DataFrame(length_distributions)
    # df = df.sorted()
    # print(df)
    df = df.fillna(0.5)
    df['length'] = df.index
    sorted_df = sorted([row for row in df.itertuples()], key=lambda x: x.length)
    df2 = pd.DataFrame(sorted_df)
    # print([x.length for x in sorted_df])
    print(df2)
    return df

def to_tsv(out_path, df):
    for dataset in datasets:
        path = Path(out_path, dataset)
        path.mkdir(exist_ok=True, parents=True)
        df[['length', dataset]].to_csv(path_or_buf=Path(path, 'length_distribution.tsv') , sep='\t', index=False)


if __name__ == '__main__':
    out_path = 'analysis/datasets'
    df = analyse_datasets('data')
    to_tsv(out_path, df)
