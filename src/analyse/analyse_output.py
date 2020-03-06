import pandas as pd
import numpy as np

from pathlib import Path
from RNA import fold
from distance import hamming


class ExperimentGroupProcessor(object):
    """TODO
    """

    def __init__(self, experiment_group_path, data_dir, output_dir, desired_gc=0.1, gc_tolerance=0.01):
        self._experiment_group = experiment_group_path
        self._data_dir = data_dir
        self._output_dir = output_dir
        self._desired_gc = desired_gc
        self._gc_tolerance = gc_tolerance

    def read_datasets(self):
        self._datasets = [dataset.resolve() for dataset in Path(self._experiment_group).glob('*/')]

    def read_methods(self):
        self._methods_raw = {}  # stores all methods of all datasets (keys)

        for dataset in self._datasets:
            methods_list = [method.resolve() for method in dataset.glob('*/')]
            self._methods_raw[dataset.name] = methods_list

    def get_method_objects(self):
        """
        Generate for each method a method object that contains and handles all results.
        """
        self._method_objects = {}
        for dataset in self._methods_raw:
            self._method_objects[dataset] = [MethodResults(method.resolve(), self._data_dir, self._desired_gc, self._gc_tolerance) for method in self._methods_raw[dataset]]

    def generate_csvs(self):
        for dataset in self._methods_raw:
            for method_object in self._method_objects[dataset]:
                method_object.df_to_csv(self._output_dir + '/' + self._experiment_group.name)

    def generate_overview(self):
        for dataset in self._methods_raw:
            for method_object in self._method_objects[dataset]:
                print(method_object.overview)


    def plot_data(self):
        for method_object in self._method_objects['rfam_taneda']:
            if 'LEARNA' in method_object._path.name:
                method_object.plot_data()

    # @property
    # def overview(self):
    #     return self._overview


class MethodResults(object):
    """TODO
    """

    def __init__(self, method_path, data_dir, desired_gc, gc_tolerance):
        self._path = method_path
        self._data_dir = data_dir
        self._dataset = Path(self._data_dir, self._path.parent.name).glob('*.rna')
        self._desired_gc = desired_gc
        self._gc_tolerance = gc_tolerance
        self.get_output()
        self.gc_over_time()
        self.constraints_and_times_to_df()


    def read_output(self):
        return self._path.glob('run-*/*.out')


    def get_output(self):
        """
        compare all output of all methods to dataset via datadir
        Hamming distance = 0 + gc_content satisfied
        """
        self._output = {}
        # for target in self._dataset:
        #     self._output[target.stem] = []
        for results_file in self.read_output():
            lines = [line.rstrip('\n') for line in open(results_file.resolve())]
            self._output[results_file.stem + '_' + results_file.parent.name] = lines

        # check results
        # valid_results = self.validate_output()
        # if not valid_results:
        #     print(self._path.name, 'something went wrong')



    def validate_output(self):
        """
        I changed the keys for output dictionary, so this won't work anymore...
        """
        validation_results = []
        if 'LEARNA' in self._path.name:
            for output in self._output:
                if self._output[output] == []:
                    continue
                validation = [hamming(fold(run[-1:][0].split()[-1:][0])[0], Path(self._data_dir, self._path.parent.name, output + '.rna').read_text().rstrip()) == 0 and self.validate_gc(run[-1:][0].split()[-1:][0]) for run in self._output[output]]
                validation_results.append(all(validation))
                print([run[-1:][0].split()[-1:][0] for run in self._output[output]])
        elif 'mcts' in self._path.name:
            for output in self._output:
                if self._output[output] == []:
                    continue
                validation = [hamming(fold(run[2][9:])[0], Path(self._data_dir, self._path.parent.name, output + '.rna').read_text().rstrip()) == 0 and self.validate_gc(run[2][9:]) for run in self._output[output]]
                validation_results.append(all(validation))
        elif 'antarna' in self._path.name:
            for output in self._output:
                if self._output[output] == []:
                    continue
                validation = [hamming(fold(run[-2:-1][0])[0], Path(self._data_dir, self._path.parent.name, output + '.rna').read_text().rstrip()) == 0 and self.validate_gc(run[-2:-1][0]) for run in self._output[output]]
                validation_results.append(all(validation))
        return any(validation_results)

    def validate_gc(self, sequence):
        """
        I changed the keys for output dictionary, so this won't work anymore...
        """
        return (self._desired_gc - self._gc_tolerance) <= (str(sequence).upper().count('G') + str(sequence).upper().count('C')) / len(str(sequence)) <= (self._desired_gc + self._gc_tolerance)

    def gc_over_time(self):
        """
        read all gc contents of all algorithms output and compare to desired_gc
        """
        self._constraints_and_times = {}
        if 'LEARNA' in self._path.name:
            for output in self._output:
                self._constraints_and_times[output] = []
                # self._agent_gcs_and_times[output] = []
                # if self._output[output] == []:
                #     continue
                for episode in self._output[output]:
                    episode_list = episode.split()
                    self._constraints_and_times[output].append((episode_list[0], episode_list[1], episode_list[2], episode_list[4], episode_list[5]))  # time, reward, fract. hamming, gc_satisfied, gc, agent_gc

    def constraints_and_times_to_df(self):
        self._constraint_dfs = {}
        for id in self._constraints_and_times:
            # if self._agent_gcs_and_times[id] ==[]:
            #     continue
            self._constraint_dfs[id] = pd.DataFrame(data=self._constraints_and_times[id], columns=['time', 'reward', 'hamming', 'gc', 'agent_gc'])
            # self._gc_dfs[id] = pd.DataFrame(data=self._constraints_and_times[id], columns=['time', 'gc'])

    def df_to_csv(self, output_path):
        Path(output_path, self._path.parent.name, self._path.name).mkdir(parents=True, exist_ok=True)
        # run_overview = pd.DataFrame(data=self._constraint_dfs)
        # print(run_overview)
        for id in self._constraints_and_times:
            self._constraint_dfs[id].to_csv(Path(output_path, self._path.parent.name, self._path.name, str(id) + '_constraints.tsv' ), sep='\t', mode='w+', index=False)
            self._constraint_dfs[id].to_csv(Path(output_path, self._path.parent.name, self._path.name, str(id) + '_reward.tsv' ), sep='\t', mode='w+', columns=['time', 'reward'], index=False)
            self._constraint_dfs[id].to_csv(Path(output_path, self._path.parent.name, self._path.name, str(id) + '_hamming.tsv' ), sep='\t', mode='w+', columns=['time', 'hamming'], index=False)
            self._constraint_dfs[id].to_csv(Path(output_path, self._path.parent.name, self._path.name, str(id) + '_gc.tsv' ), sep='\t', mode='w+', columns=['time', 'gc'], index=False)
            self._constraint_dfs[id].to_csv(Path(output_path, self._path.parent.name, self._path.name, str(id) + '_agent_gc.tsv' ), sep='\t', mode='w+', columns=['time', 'agent_gc'], index=False)
            # self._gc_dfs[id].to_csv(Path(output_path, self._path.parent.name, self._path.name, str(id) + '_gc.tsv' ), sep='\t', mode='w+', index=False)

    def generate_overview_df(self):
        #overview = pd.DataFrame()
        #for id in self._constraint_dfs:
        #    overview.append(df, )
        #return pd.DataFrame(data=self._constraint_dfs, index=[key for key in self._constraint_dfs])

        return pd.concat([self._constraint_dfs[id] for id in self._constraint_dfs if not self._constraint_dfs[id].bool], keys=[id for id in self._constraint_dfs if not self._constraint_dfs[id].bool])
        # for df in self._constraint_dfs.values():
        #     print(df)




    def plot_data(self):
        pass

    @property
    def dfs(self):
        return self._constraint_dfs

    @property
    def overview(self):
        return self.generate_overview_df()





if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument(
        "--experiment_group", type=Path, help="Path to experiment results"
    )
    parser.add_argument("--data_dir", default="data", help="Data directory")
    parser.add_argument("--out_dir", default="analysis", help="Output directory")

    # parser.add_argument("--dataset", type=Path, help="Available: eterna, rfam_taneda")
    parser.add_argument("--desired_gc", default=0.1, type=float, help="The desired GC-content of the sequences")

    parser.add_argument("--gc_tolerance", default=0.01, type=float, help="The tolerance of GC-content")


    args = parser.parse_args()


    results = ExperimentGroupProcessor(args.experiment_group, args.data_dir, args.out_dir, args.desired_gc, args.gc_tolerance)

    results.read_datasets()
    results.read_methods()
    results.get_method_objects()
    results.generate_csvs()
    # results.plot_data()
    # results.generate_overview()
