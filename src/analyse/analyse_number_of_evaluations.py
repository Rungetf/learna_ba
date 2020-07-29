"""
    return: Results {method: {dataset: {task_id: evaluations}}}
"""

import numpy as np

from pathlib import Path
from RNA import fold
from distance import hamming
from collections import defaultdict

def read_results(path):
    return Path(path).glob('*.out')

def analyse(path, methods, data_dir, out_dir):
    results = dict()  # defaultdict(defaultdict(dict()))

    for i in methods:
        print(f"Starting with method {i}")
        # print(f"start counting number of evaluations for {mapping_method_to_realname[i]}")
        paths = Path(path).glob(f"**/{i}/*/*.out")
        evaluations = analyse_method(paths, data_dir)
        results[i] = evaluations
        # print(results)

    analysis_per_length = analyse_results_per_length(results, data_dir)

def analyse_method(paths, data_dir):  # , method_names):
    # non_output_matching_files = []
    # sequence_length_counts = {}
    method_evaluations_per_testset = defaultdict(dict)
    for p in paths:
        # print(f"Start analyzing {p}")

        output = p.read_text().rstrip().split(sep='\n')

        try:
            method_evaluations_per_testset[p.parent.parent.parent.name][p.stem].append(len(output) - 2)
        except KeyError:
            method_evaluations_per_testset[p.parent.parent.parent.name][p.stem] = [len(output) - 2]

    return method_evaluations_per_testset

def analyse_results_per_length(results, data_dir):
    eterna_evaluation_counter_per_length = defaultdict(dict)
    eterna_avg = defaultdict(dict)
    for method in results:
        if method == '7052569_471_0_8':
            for dataset in results[method]:
                for id in results[method][dataset]:
                    if dataset[-1] == 'c':
                        continue
                    data_path = Path(data_dir, 'rfam_processed', dataset, f"{id}.rna")
                    if dataset == 'eterna':
                        data_path = Path(data_dir, dataset, f"{id}.rna")
                    target = data_path.read_text().rstrip()
                    if dataset == 'eterna':
                        length = len(target)
                    else:
                        print(target)
                    avg = np.mean(results[method][dataset][id])
                    eterna_avg[id][length] = avg
                    eterna_evaluation_counter_per_length[length] = results[method][dataset][id]
                    # print(dataset, id)
    avg_values = []
    for id in eterna_avg:
        for length in eterna_avg[id]:
            avg_values.append((id, length, eterna_avg[id][length]))
            print(id, length, eterna_avg[id][length])

    acc_length = sum([x[1] for x in avg_values])
    acc_evals = sum([x[2] for x in avg_values])

    print(f"autoMeta-LEARNA avg evaluations per nucleotide: {acc_evals / acc_length}")





if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument(
        "--experiment_group", type=Path, help="Path to experiment results"
    )
    parser.add_argument("--methods", type=str, default=["7052569_471_0_8"], nargs="+", help="The method's experiment_id that should be included for analysis")
    parser.add_argument("--data_dir", default="data", help="Data directory")
    parser.add_argument("--out_dir", default="analysis", help="Output directory")

    args = parser.parse_args()
    # methods = ['7052569_471_0_8', '7052570_212_0_0', '7052571_188_0_3']

    path = args.experiment_group.resolve()

    # mapping_method_to_realname = {'7052569_471_0_8': 'autoMeta-LEARNA', '7052570_212_0_0': 'autoMeta-LEARNA-Adapt', '7052571_188_0_3': 'autoLEARNA'}
    out_path = Path(args.out_dir, args.experiment_group)
    out_path.mkdir(parents=True, exist_ok=True)

    data = Path(args.data_dir)

    methods = args.methods

    analyse(path, methods, data, out_path)

    # for i in methods:
    #     # print(f"start counting number of evaluations for {mapping_method_to_realname[i]}")
    #     paths = Path(path).glob(f"**/{i}/*/*.out")
    #     analyse_evaluations(paths, mapping_method_to_realname)
    #     #     print(p)





    # path = Path('results', 'partial_rna_design', 'rfam_anta_sc', '7052569_471_0_8', 'run-0')
    # out_path = Path('verification/partial_rna_design/rfam_anta_sc/7052569_471_0_8/run-0')
    # out_path.mkdir(parents=True, exist_ok=True)
    # print('start')
    # designs, foldings, ids = verify(path)
    # print(foldings)
    # print('got everything')
    # target_structures = []
    # target_sequences = []
    # for i, f, d in zip(ids, foldings, designs):
    #     structure = Path('data', 'rfam_anta_sc', f"{i}.rna").read_text().rstrip().split()[1]
    #     sequence  = Path('data', 'rfam_anta_sc', f"{i}.rna").read_text().rstrip().split()[2]
    #     target  = Path('data', 'rfam_anta_sc', f"{i}.rna").read_text().rstrip().split()[3]
    #     print(i)
    #     print(f"t: {target}")
    #     print(f"d: {f}")
    #     print(f"t: {structure}")
    #     print(f"d: {d}")
    #     print(f"t: {sequence}")
    #     target_structures.append(structure)
    #     target_sequences.append(sequence)
    # print('starting writting')
    # write_verifications(out_path, ids, designs, foldings, target_structures, target_sequences)
