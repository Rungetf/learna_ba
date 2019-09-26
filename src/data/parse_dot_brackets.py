import pandas as pd
from pathlib import Path


def parse_dot_brackets(
    dataset, data_dir, target_structure_ids=None, target_structure_path=None
):
    """TODO
    Generate the targets for next epoch.

    Args:
        dataset: The name of the benchmark to use targets from.
        data_dir: The directory of the target structures.
        target_structure_ids: Use specific targets by ids.
        target path: Specify a path to the targets.

    Returns:
        An epoch generator for the specified target structures.
    """
    if target_structure_path:
        target_paths = [target_structure_path]
    elif target_structure_ids:
        target_paths = [
            Path(data_dir, dataset, f"{id_}.rna") for id_ in target_structure_ids
        ]
    else:
        target_paths = list(Path(data_dir, dataset).glob("*.rna"))

    return [data_path.read_text().rstrip() for data_path in target_paths]

def parse_local_design_data(
    dataset, data_dir, target_structure_ids=None, target_structure_path=None
):
    """TODO
    Generate targets for next local design epoch.
    """
    path = Path
    df = pd.read_csv(Path(data_dir, dataset, f"{str(dataset).split('_')[-1]}.interim"), sep='\t')
    if target_structure_ids:
        # return list(map(tuple, df.iloc[[int(id) for id in target_structure_ids]][['structure', 'sequence', 'local_random', 'local_motif', 'gc_content', 'mfe']].itertuples(index=False, name=None)))
        # for row in zip(df.iloc[[int(id) for id in target_structure_ids]]['structure'], df.iloc[[int(id) for id in target_structure_ids]]['sequence'], df.iloc[[int(id) for id in target_structure_ids]]['local_random'], df.iloc[[int(id) for id in target_structure_ids]]['local_motif'], df.iloc[[int(id) for id in target_structure_ids]]['gc_content'], df.iloc[[int(id) for id in target_structure_ids]]['mfe']):
        #     print(row[0])
        #     print(row[1])
        #     print(row[2])
        #     print(row[3])
        #     print(row[4])
        #     print(row[5])
        for id in target_structure_ids:
            df.iloc[int(id)]

        return [(index+1, row[0], row[1], row[2], row[3], row[4], row[5]) for index, row in enumerate(zip(df.iloc[[int(id) for id in target_structure_ids]]['structure'], df.iloc[[int(id) for id in target_structure_ids]]['sequence'], df.iloc[[int(id) for id in target_structure_ids]]['local_random'], df.iloc[[int(id) for id in target_structure_ids]]['local_motif'], df.iloc[[int(id) for id in target_structure_ids]]['gc_content'], df.iloc[[int(id) for id in target_structure_ids]]['mfe']))]
    elif target_structure_path:
        return [tuple(Path(target_structure_path).read_text().rstrip().split())]
    else:
        # return list(map(tuple, df[['structure', 'sequence', 'local_random', 'local_motif', 'gc_content', 'mfe']].itertuples(index=False, name=None)))
        return [(index+1, row[0], row[1], row[2], row[3], row[4], row[5]) for index, row in enumerate(zip(df['structure'], df['sequence'], df['local_random'], df['local_motif'], df['gc_content'], df['mfe']))]
