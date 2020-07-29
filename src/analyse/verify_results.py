from pathlib import Path
from RNA import fold
from distance import hamming

def read_results(path):
    return Path(path).glob('*.out')

def verify(path):
    designs = []
    foldings = []
    targets = []
    ids = []
    for r in read_results(path):
        id = r.stem
        design = r.read_text().rstrip().split()[-1]
        designs.append(design)
        folding = fold(design)[0]
        foldings.append(folding)
        # targets.append(Path(f"data/rfam_local_test/{id}.rna").read_text().rstrip().split()[3])
        ids.append(id)
        # distances.append(hamming(folding, target))
    return designs, foldings, ids  # targets, ids

def write_verifications(out_path, ids, designs, foldings, structures, sequences):
    for id, folding, design, structure, sequence in zip(ids, foldings, designs, structures, sequences):
        f = Path(out_path, f"{id}.verify")
        # f.mkdir(parents=True, exist_ok=True)
        f.write_text(str(folding) + "\n" + str(design) + "\n" + str(structure) + "\n" + str(sequence))


        # path = str(out_path.as_posix() + f"/{id}.verify")

        #
        # Path(path).mkdir(parents=True, exist_ok=True)q
        # with open(path, 'a+') as f:
        # # out_path = Path(out_path, f"{id}.verify")
        # # out_path.write_text(f"{target}\t{folding}\t{design}")
        #     f.write(target)
        #     f.write(folding)
        #     f.write(design)




if __name__ == '__main__':
    path = Path('results', 'partial_rna_design', 'rfam_anta_sc', '7052569_471_0_8', 'run-0')
    out_path = Path('verification/partial_rna_design/rfam_anta_sc/7052569_471_0_8/run-0')
    out_path.mkdir(parents=True, exist_ok=True)
    print('start')
    designs, foldings, ids = verify(path)
    print(foldings)
    print('got everything')
    target_structures = []
    target_sequences = []
    for i, f, d in zip(ids, foldings, designs):
        structure = Path('data', 'rfam_anta_sc', f"{i}.rna").read_text().rstrip().split()[1]
        sequence  = Path('data', 'rfam_anta_sc', f"{i}.rna").read_text().rstrip().split()[2]
        target  = Path('data', 'rfam_anta_sc', f"{i}.rna").read_text().rstrip().split()[3]
        print(i)
        print(f"t: {target}")
        print(f"d: {f}")
        print(f"t: {structure}")
        print(f"d: {d}")
        print(f"t: {sequence}")
        target_structures.append(structure)
        target_sequences.append(sequence)
    print('starting writting')
    write_verifications(out_path, ids, designs, foldings, target_structures, target_sequences)
