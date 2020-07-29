from pathlib import Path
from distance import hamming

from RNA import fold


def get_results(path):
    return Path(path).glob('*.out')

def validate_freinet(path):
    pass

if __name__ == '__main__':
    results_path = Path('results', 'rna_local_design', 'rfam_taneda_local_test')
    dataset_path = Path('data', 'rfam_taneda')
    runs = 1
    valid = []

    for method in results_path.glob('*/'):
        print(method.stem)
        if str(method.stem) == '6782534_414_0_5_output':
            print(f"Enter internal loop for {method.stem}")
            for i in range(runs):
                print(i)
                for path in get_results(Path(results_path, method, f"run-{i}/")):
                    result = path.read_text().rstrip().split('\t')[3]
                    print(path)
                    print(result)
                    valid.append(hamming(fold(result)[0], Path(dataset_path, f"{path.stem}.rna").read_text()) == 0)
    print(all(valid))
