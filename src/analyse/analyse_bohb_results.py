import numpy as np
import matplotlib.pyplot as plt
import hpbandster.core.result as hpres
import hpbandster.visualization as hpvis
from src.optimization.learna_worker import LearnaWorker
from src.optimization.meta_learna_worker import MetaLearnaWorker
from fanova import fANOVA
import fanova.visualizer
from pathlib import Path
import ConfigSpace as CS
import itertools as it


def get_meta_freinet_config():
    config_space = CS.ConfigurationSpace()

    # parameters for PPO here
    config_space.add_hyperparameter(
        CS.UniformFloatHyperparameter(
            "learning_rate", lower=1e-6, upper=1e-3, log=True, default_value=5e-4  # FR: changed learning rate lower from 1e-5 to 1e-6, ICLR: Learna (5,99e-4), Meta-LEARNA (6.44e-5)
        )
    )
    config_space.add_hyperparameter(
        CS.UniformIntegerHyperparameter(
            "batch_size", lower=32, upper=256, log=True, default_value=32  # FR: changed batch size upper from 128 to 256, configs from ICLR used 126 (LEARNA) and 123 (Meta-LEARNA)
        )
    )
    config_space.add_hyperparameter(
        CS.UniformFloatHyperparameter(
            "entropy_regularization",
            lower=1e-7,  # FR: changed entropy regularization lower from 1e-5 to 1e-7, ICLR: LEARNA (6,76e-5), Meta-LEARNA (151e-4)
            upper=1e-2,
            log=True,
            default_value=1.5e-3,
        )
    )

    config_space.add_hyperparameter(
        CS.UniformFloatHyperparameter(
            "reward_exponent", lower=1, upper=12, default_value=1  # FR: changed reward_exponent upper from 10 to 12, ICLR: Learna (9.34), Meta-LEARNA (8.93)
        )
    )

    config_space.add_hyperparameter(
        CS.UniformFloatHyperparameter(
            "state_radius_relative", lower=0, upper=1, default_value=0
        )
    )

    # parameters for the architecture
    config_space.add_hyperparameter(
        CS.UniformIntegerHyperparameter(
            "conv_radius1", lower=0, upper=8, default_value=1
        )
    )
    config_space.add_hyperparameter(
        CS.UniformIntegerHyperparameter(
            "conv_channels1", lower=1, upper=32, log=True, default_value=32
        )
    )

    config_space.add_hyperparameter(
        CS.UniformIntegerHyperparameter(
            "conv_radius2", lower=0, upper=4, default_value=0
        )
    )
    config_space.add_hyperparameter(
        CS.UniformIntegerHyperparameter(
            "conv_channels2", lower=1, upper=32, log=True, default_value=1
        )
    )

    config_space.add_hyperparameter(
        CS.UniformIntegerHyperparameter(
            "num_fc_layers", lower=1, upper=2, default_value=2
        )
    )
    config_space.add_hyperparameter(
        CS.UniformIntegerHyperparameter(
            "fc_units", lower=8, upper=64, log=True, default_value=50
        )
    )

    config_space.add_hyperparameter(
        CS.UniformIntegerHyperparameter(
            "num_lstm_layers", lower=0, upper=3, default_value=0  # FR: changed lstm layers upper from 2 to 3
        )
    )
    config_space.add_hyperparameter(
        CS.UniformIntegerHyperparameter(
            "lstm_units", lower=1, upper=64, log=True, default_value=1
        )
    )

    config_space.add_hyperparameter(
        CS.UniformIntegerHyperparameter(
            "embedding_size", lower=0, upper=9, default_value=1  # FR: changed embedding size upper from 4 to 8
        )
    )

    config_space.add_hyperparameter(
        CS.CategoricalHyperparameter(
            "reward_function", choices=['sequence_and_structure', 'structure_replace_sequence', 'structure_only']
        )
    )

    config_space.add_hyperparameter(
        CS.CategoricalHyperparameter(
            "state_representation", choices=['n-gram', 'sequence_progress']
        )
    )

    config_space.add_hyperparameter(
        CS.CategoricalHyperparameter(
            "trainingset", choices=['rfam_local_short_train', 'rfam_local_train', 'rfam_local_long_train']
        )
    )

    config_space.add_hyperparameter(
        CS.CategoricalHyperparameter(
            "data_type", choices=['random', 'random-sort']
        )
    )

    config_space.add_hyperparameter(
        CS.UniformIntegerHyperparameter(
            "predict_pairs", lower=0, upper=1, default_value=1
        )
    )

    return config_space


def get_freinet_config():
    config_space = CS.ConfigurationSpace()
    # parameters for PPO here
    config_space.add_hyperparameter(
        CS.UniformFloatHyperparameter(
            "learning_rate", lower=1e-6, upper=1e-3, log=True, default_value=5e-4  # FR: changed learning rate lower from 1e-5 to 1e-6, ICLR: Learna (5,99e-4), Meta-LEARNA (6.44e-5)
        )
    )
    config_space.add_hyperparameter(
        CS.UniformIntegerHyperparameter(
            "batch_size", lower=32, upper=256, log=True, default_value=32  # FR: changed batch size upper from 128 to 256, configs from ICLR used 126 (LEARNA) and 123 (Meta-LEARNA)
        )
    )
    config_space.add_hyperparameter(
        CS.UniformFloatHyperparameter(
            "entropy_regularization",
            lower=1e-7,  # FR: changed entropy regularization lower from 1e-5 to 1e-7, ICLR: LEARNA (6,76e-5), Meta-LEARNA (151e-4)
            upper=1e-2,
            log=True,
            default_value=1.5e-3,
        )
    )

    config_space.add_hyperparameter(
        CS.UniformFloatHyperparameter(
            "reward_exponent", lower=1, upper=12, default_value=1  # FR: changed reward_exponent upper from 10 to 12, ICLR: Learna (9.34), Meta-LEARNA (8.93)
        )
    )

    config_space.add_hyperparameter(
        CS.UniformFloatHyperparameter(
            "state_radius_relative", lower=0, upper=1, default_value=0
        )
    )

    # parameters for the architecture
    config_space.add_hyperparameter(
        CS.UniformIntegerHyperparameter(
            "conv_radius1", lower=0, upper=8, default_value=1
        )
    )
    config_space.add_hyperparameter(
        CS.UniformIntegerHyperparameter(
            "conv_channels1", lower=1, upper=32, log=True, default_value=32
        )
    )

    config_space.add_hyperparameter(
        CS.UniformIntegerHyperparameter(
            "conv_radius2", lower=0, upper=4, default_value=0
        )
    )
    config_space.add_hyperparameter(
        CS.UniformIntegerHyperparameter(
            "conv_channels2", lower=1, upper=32, log=True, default_value=1
        )
    )

    config_space.add_hyperparameter(
        CS.UniformIntegerHyperparameter(
            "num_fc_layers", lower=1, upper=2, default_value=2
        )
    )
    config_space.add_hyperparameter(
        CS.UniformIntegerHyperparameter(
            "fc_units", lower=8, upper=64, log=True, default_value=50
        )
    )

    config_space.add_hyperparameter(
        CS.UniformIntegerHyperparameter(
            "num_lstm_layers", lower=0, upper=3, default_value=0  # FR: changed lstm layers upper from 2 to 3
        )
    )
    config_space.add_hyperparameter(
        CS.UniformIntegerHyperparameter(
            "lstm_units", lower=1, upper=64, log=True, default_value=1
        )
    )

    config_space.add_hyperparameter(
        CS.UniformIntegerHyperparameter(
            "embedding_size", lower=0, upper=9, default_value=1  # FR: changed embedding size upper from 4 to 8
        )
    )

    # config_space.add_hyperparameter(
    #     CS.UniformIntegerHyperparameter(
    #         "sequence_reward", lower=0, upper=1, default_value=0
    #     )
    # )

    config_space.add_hyperparameter(
        CS.CategoricalHyperparameter(
            "reward_function", choices=['sequence_and_structure', 'structure_replace_sequence', 'structure_only']
        )
    )

    config_space.add_hyperparameter(
        CS.CategoricalHyperparameter(
            "state_representation", choices=['n-gram', 'sequence_progress']
        )
    )

    config_space.add_hyperparameter(
        CS.CategoricalHyperparameter(
            "data_type", choices=['random', 'random-sort']
        )
    )


    config_space.add_hyperparameter(
        CS.UniformIntegerHyperparameter(
            "predict_pairs", lower=0, upper=1, default_value=1
        )
    )


    # config_space.add_hyperparameter(
    #     CS.UniformFloatHyperparameter(
    #         "structural_weight", lower=0, upper=1, default_value=1
    #     )
    # )

    # config_space.add_hyperparameter(
    #     CS.UniformFloatHyperparameter(
    #         "gc_weight", lower=0, upper=1, default_value=1
    #     )
    # )


    return config_space

#     config_space.add_hyperparameter(
#         CS.UniformFloatHyperparameter(
#             "reward_exponent", lower=1, upper=12, default_value=1  # FR: changed reward_exponent upper from 10 to 12, ICLR: Learna (9.34), Meta-LEARNA (8.93)
#         )
#     )
# 
#     config_space.add_hyperparameter(
#         CS.UniformFloatHyperparameter(
#             "state_radius_relative", lower=0, upper=1, default_value=0
#         )
#     )
# 
#     # parameters for the architecture
#     config_space.add_hyperparameter(
#         CS.UniformIntegerHyperparameter(
#             "conv_radius1", lower=0, upper=8, default_value=1
#         )
#     )
#     config_space.add_hyperparameter(
#         CS.UniformIntegerHyperparameter(
#             "conv_channels1", lower=1, upper=32, log=True, default_value=32
#         )
#     )
# 
#     config_space.add_hyperparameter(
#         CS.UniformIntegerHyperparameter(
#             "conv_radius2", lower=0, upper=4, default_value=0
#         )
#     )
#     config_space.add_hyperparameter(
#         CS.UniformIntegerHyperparameter(
#             "conv_channels2", lower=1, upper=32, log=True, default_value=1
#         )
#     )


def get_fine_tuning_config():
    config_space = CS.ConfigurationSpace()

    # parameters for PPO here
    config_space.add_hyperparameter(
        CS.UniformFloatHyperparameter(
            "learning_rate", lower=1e-6, upper=1e-3, log=True, default_value=5e-4  # FR: changed learning rate lower from 1e-5 to 1e-6, ICLR: Learna (5,99e-4), Meta-LEARNA (6.44e-5)
        )
    )
    config_space.add_hyperparameter(
        CS.UniformIntegerHyperparameter(
            "batch_size", lower=32, upper=256, log=True, default_value=32  # FR: changed batch size upper from 128 to 256, configs from ICLR used 126 (LEARNA) and 123 (Meta-LEARNA)
        )
    )
    config_space.add_hyperparameter(
        CS.UniformFloatHyperparameter(
            "entropy_regularization",
            lower=1e-7,  # FR: changed entropy regularization lower from 1e-5 to 1e-7, ICLR: LEARNA (6,76e-5), Meta-LEARNA (151e-4)
            upper=1e-2,
            log=True,
            default_value=1.5e-3,
        )
    )

    config_space.add_hyperparameter(
        CS.UniformFloatHyperparameter(
            "reward_exponent", lower=1, upper=12, default_value=1  # FR: changed reward_exponent upper from 10 to 12, ICLR: Learna (9.34), Meta-LEARNA (8.93)
        )
    )

    config_space.add_hyperparameter(
        CS.UniformFloatHyperparameter(
            "state_radius_relative", lower=0, upper=1, default_value=0
        )
    )

    # parameters for the architecture
    config_space.add_hyperparameter(
        CS.UniformIntegerHyperparameter(
            "conv_radius1", lower=0, upper=8, default_value=1
        )
    )
    config_space.add_hyperparameter(
        CS.UniformIntegerHyperparameter(
            "conv_channels1", lower=1, upper=32, log=True, default_value=32
        )
    )

    config_space.add_hyperparameter(
        CS.UniformIntegerHyperparameter(
            "conv_radius2", lower=0, upper=4, default_value=0
        )
    )
    config_space.add_hyperparameter(
        CS.UniformIntegerHyperparameter(
            "conv_channels2", lower=1, upper=32, log=True, default_value=1
        )
    )

    config_space.add_hyperparameter(
        CS.UniformIntegerHyperparameter(
            "num_fc_layers", lower=1, upper=2, default_value=2
        )
    )
    config_space.add_hyperparameter(
        CS.UniformIntegerHyperparameter(
            "fc_units", lower=8, upper=64, log=True, default_value=50
        )
    )

    config_space.add_hyperparameter(
        CS.UniformIntegerHyperparameter(
            "num_lstm_layers", lower=0, upper=3, default_value=0  # FR: changed lstm layers upper from 2 to 3
        )
    )
    config_space.add_hyperparameter(
        CS.UniformIntegerHyperparameter(
            "lstm_units", lower=1, upper=64, log=True, default_value=1
        )
    )

    config_space.add_hyperparameter(
        CS.UniformIntegerHyperparameter(
            "embedding_size", lower=0, upper=9, default_value=1  # FR: changed embedding size upper from 4 to 8
        )
    )

    config_space.add_hyperparameter(
        CS.CategoricalHyperparameter(
            "state_representation", choices=['n-gram', 'sequence_progress']
        )
    )

    config_space.add_hyperparameter(
        CS.CategoricalHyperparameter(
            "trainingset", choices=['rfam_local_short_train', 'rfam_local_train', 'rfam_local_long_train']
        )
    )

    config_space.add_hyperparameter(
        CS.CategoricalHyperparameter(
            "data_type", choices=['random', 'random-sort']
        )
    )

    config_space.add_hyperparameter(
        CS.CategoricalHyperparameter(
            "state_representation", choices=['n-gram', 'sequence_progress']
        )
    )

    config_space.add_hyperparameter(
        CS.CategoricalHyperparameter(
            "trainingset", choices=['rfam_local_short_train', 'rfam_local_train', 'rfam_local_long_train']
        )
    )

    config_space.add_hyperparameter(
        CS.CategoricalHyperparameter(
            "data_type", choices=['random', 'random-sort']
        )
    )

    return config_space



def analyse_bohb_run(run):
    # load the example run from the log files
    result = hpres.logged_results_to_HBS_result(f"results/bohb/{run}")

    # get all executed runs
    all_runs = result.get_all_runs()

    # get the 'dict' that translates config ids to the actual configurations
    id2conf = result.get_id2config_mapping()


    # Here is how you get he incumbent (best configuration)
    inc_id = result.get_incumbent_id()

    # let's grab the run on the highest budget
    inc_runs = result.get_runs_by_id(inc_id)
    inc_run = inc_runs[-1]


    # We have access to all information: the config, the loss observed during
    #optimization, and all the additional information
    inc_loss = inc_run.loss
    inc_config = id2conf[inc_id]['config']
    inc_test_loss = inc_run.info['validation_info']

    # print(f"validation info of inc: {inc_test_loss}")

    # Let's plot the observed losses grouped by budget,
    # hpvis.losses_over_time(all_runs)

    # # the number of concurent runs,
    # hpvis.concurrent_runs_over_time(all_runs)

    # # and the number of finished runs.
    # hpvis.finished_runs_over_time(all_runs)

    # # This one visualizes the spearman rank correlation coefficients of the losses
    # # between different budgets.
    # hpvis.correlation_across_budgets(result)

    # # For model based optimizers, one might wonder how much the model actually helped.
    # # The next plot compares the performance of configs picked by the model vs. random ones
    # hpvis.performance_histogram_model_vs_random(all_runs, id2conf)

    # plt.show()

    # # df = result.get_pandas_dataframe()
    # # print(df)

    min_solved = 0

    all_solved = [(x.info['validation_info']['num_solved'], x.config_id, id2conf[x.config_id]['config']) for x in all_runs if x.info and int(x.info['validation_info']['num_solved']) >= min_solved]
    # # print(all_solved)
    all_solved_sorted = sorted(all_solved, key=lambda x: x[0], reverse=True)

    print(f"number of configurations evaluated: {len(all_solved)}")

    print(f"{len(all_solved_sorted)} configurations solved at least {min_solved} targets:")

    print(f"Most solving config: {all_solved_sorted[0][1]}")
    # print('batch\tc_channels1\tc_channels2\tc_radius1\tc_radius2\tembedding\tentropy\tfc_units\tlearning_rate\tlstm_units\tfc_layers\tlstm_layers\tpairs\talpha\treward_f\ts_radius')
    # for index, i in enumerate(all_solved_sorted[:10]):
    for index, i in enumerate(all_solved_sorted[:10]):
        print(f"[{index + 1}]")
        print(str(i) + '\n')
        # print(f"{i[2]['batch_size']}\t{i[2]['conv_channels1']}\t{i[2]['conv_channels2']}\t{i[2]['conv_radius1']}\t{i[2]['conv_radius2']}\t{i[2]['embedding_size']}\t{i[2]['entropy_regularization']}\t{i[2]['fc_units']}\t{i[2]['learning_rate']}\t{i[2]['lstm_units']}\t{i[2]['num_fc_layers']}\t{i[2]['num_lstm_layers']}\t{i[2]['predict_pairs']}\t{i[2]['reward_exponent']}\t{i[2]['reward_function']}\t{i[2]['state_radius_relative']}")

    # print(all_solved)
    # print('\n')
    # print('Incumbent:')
    # print(inc_config)
    # print('It achieved accuracies of %f (validation) and %f (test).'%(1-inc_loss, inc_test_loss))




def generate_fanova_plots(path, run, out_dir, mode, n, param):
    # load the example run from the log files
    result = hpres.logged_results_to_HBS_result(f"{path}/{run}")

    # param = "reward_function"

    if mode == 'autoLEARNA':
        cs = get_freinet_config()
    elif mode == 'autoMeta-LEARNA':
        cs = get_meta_freinet_config()
    else:
        raise
    print('generate fanova data')
    a, b, _ = result.get_fANOVA_data(cs)
    b = np.array([100 - np.float64(x) for x in b])
    print('create fanova object')
    f = fANOVA(a, b, cs)
    print('create visualizer')
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    vis = fanova.visualizer.Visualizer(f, cs, directory=path)
    hp_importances = []
    print('computing importance of parameters')
    for i in range(1, len(cs.get_hyperparameter_names())):
        print(f"Computing importance of {cs.get_hyperparameter_by_idx(i)}")
        importance = f.quantify_importance(dims=[cs.get_hyperparameter_by_idx(i)])[(cs.get_hyperparameter_by_idx(i), )]
        print(f"total importance: {importance['total importance']}")
        hp_name = cs.get_hyperparameter_by_idx(i)
        # print(hp_name)
        total_importance = importance['total importance']
        # print(total_importance)
        individual_importance = importance['individual importance']
        # print(individual_importance)
        individual_std = importance['individual std']
        # print(individual_std)
        total_std = importance['total std']
        # print(total_importance)
        # print('\n')
        hp_importances.append((hp_name, total_importance, total_std, individual_importance, individual_std))
        # print(hp_importances)
        print(f"generate marginal plot for hyperparameter: {cs.get_hyperparameter_by_idx(i)}")
        try:
            log = cs.get_hyperparameter(cs.get_hyperparameter_by_idx(i)).log
        except:
            log = False
        # print(f"log_scale is {log}")
        fig = vis.plot_marginal(param=i, log_scale=log, show=False)
        fig.savefig(Path('results', 'fanova', run, f"{cs.get_hyperparameter_by_idx(i)}"))
        fig.close()
    hp_importances = sorted(hp_importances, key=lambda x: x[1], reverse=True)
    # print(hp_importances)
    print('Writing importance files')
    for hp_importance in hp_importances:
        # print(hp_importance)
        imps_path = Path('results', 'fanova', run, 'importances.txt')
        imp_path = Path('results', 'fanova', run, f"importance_{hp_importance[0]}.txt")
        with open(imp_path, 'w+') as imp:
            # print(str(hp_importance[0])
            imp.write(str(hp_importance[0]) + '\n')
            # print(str(hp_importance[1])
            imp.write('total importance: ' + str(hp_importance[1]) + '\n')
            # print(str(hp_importance[2])
            imp.write('total_std: ' + str(hp_importance[2]) + '\n')
            # print(str(hp_importance[3])
            imp.write('individual importance: ' + str(hp_importance[3]) + '\n')
            # print(str(hp_importance[4])
            imp.write('individual std: ' + str(hp_importance[4]) + '\n')
        with open(imps_path, 'a+') as imps:
            imps.write(str(hp_importance) + '\n')
                # imps.write('\n')
    # print(importance)

        # print('generate plots')
        # vis.create_most_important_pairwise_marginal_plots(n=n)
        # for i in range(1, len(cs.get_hyperparameter_names())):
        # try:
        # except Exception as e:
        #     print(e)
    # for i in range(1, len(cs.get_hyperparameter_names())):
    #     for j in range(1, len(cs.get_hyperparameter_names())):
    #         try:
    #             if i != j:
    #                 print(f"generate pairwise marginal plot for hyperparameters: {cs.get_hyperparameter_by_idx(i)} and {cs.get_hyperparameter_by_idx(j)}")
    #                 fig = vis.plot_pairwise_marginal(param_list=(i, j), show=False)
    #                 fig.savefig(Path('results', 'fanova', run, f"pairwiswe_marginal_{cs.get_hyperparameter_by_idx(i)}_{cs.get_hyperparameter_by_idx(j)}"))
    #                 fig.close()
    #         except Exception as e:
    #             print(e)

    # # # getting the 10 most important pairwise marginals sorted by importance
    # # best_margs = f.get_most_important_pairwise_marginals(n=10)
    # # print(best_margs)
    # # # creating the plot of pairwise marginal:
    # # vis.plot_pairwise_marginal((0,2), resolution=20)
    # # creating all plots in the directory


def create_pairwise_marginals(path, run, out_dir, mode, params):
    # load the example run from the log files
    result = hpres.logged_results_to_HBS_result(f"{path}/{run}")

    # param = "reward_function"

    if mode == 'autoLEARNA':
        cs = get_freinet_config()
    elif mode == 'autoMeta-LEARNA':
        cs = get_meta_freinet_config()
    else:
        raise
    print('generate fanova data')
    a, b, _ = result.get_fANOVA_data(cs)
    b = np.array([100 - np.float64(x) for x in b])
    print('create fanova object')
    f = fANOVA(a, b, cs)
    print('create visualizer')
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    vis = fanova.visualizer.Visualizer(f, cs, directory=path)
    pairwise_parameters = it.permutations(params, 2)
    for item in pairwise_parameters:
        p1 = item[0]
        p2 = item[1]
        print(f"generate pairwise marginal plot for hyperparameters: {p1} and {p2}")
        fig = vis.plot_pairwise_marginal(param_list=(cs.get_idx_by_hyperparameter_name(p1), cs.get_idx_by_hyperparameter_name(p2)), show=False)
        fig.savefig(Path('results', 'fanova', run, f"pairwiswe_marginal_{p1}_{p2}"))
        fig.close()
    # print(f"Creating {n} most important pairwise marginal plots")
    # vis.create_most_important_pairwise_marginal_plots(f"results/fanova/{run}/", n)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--path", type=str, default="results/bohb", help="Path to results of run"
    )

    parser.add_argument(
        "--run", type=str, help="The id of the run"
    )

    parser.add_argument(
        "--out_dir", type=str, help="The id of the run"
    )

    parser.add_argument(
        "--mode", type=str, help="Choose between autoLEARNA, autoMeta-LEARNA"
    )

    parser.add_argument(
        "--n", type=int, help="The number of most important marginal plots fanova should generate"
    )

    parser.add_argument(
        "--parameter", type=str, help="The parameter to analyse"
    )

    args = parser.parse_args()

    params = ['state_radius_relative', 'learning_rate', 'num_lstm_layers']

    analyse_bohb_run(args.run)

    # generate_fanova_plots(args.path, args.run, args.out_dir, args.mode, args.n, args.parameter)

    # create_pairwise_marginals(args.path, args.run, args.out_dir, args.mode, params)
