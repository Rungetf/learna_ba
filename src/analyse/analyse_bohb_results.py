import numpy as np
import matplotlib.pyplot as plt
import hpbandster.core.result as hpres
import hpbandster.visualization as hpvis
from src.optimization.learna_worker import LearnaWorker
from fanova import fANOVA
import fanova.visualizer
from pathlib import Path


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

    # df = result.get_pandas_dataframe()
    # print(df)

    min_solved = 1

    all_solved = [(x.info['validation_info']['num_solved'], x.config_id, id2conf[x.config_id]['config']) for x in all_runs if x.info and int(x.info['validation_info']['num_solved']) >= min_solved]
    # # print(all_solved)
    all_solved_sorted = sorted(all_solved, key=lambda x: x[0], reverse=True)

    print(f"{len(all_solved_sorted)} configurations solved at least {min_solved} targets:")

    print(f"Most solving config: {all_solved_sorted[0][1]}")
    print('batch\tc_channels1\tc_channels2\tc_radius1\tc_radius2\tembedding\tentropy\tfc_units\tlearning_rate\tlstm_units\tfc_layers\tlstm_layers\tpairs\talpha\treward_f\ts_radius')
    # for index, i in enumerate(all_solved_sorted[:10]):
    for index, i in enumerate(all_solved_sorted[:10]):
        # print(f"[{index + 1}]")
        print(str(i) + '\n')
        print(f"{i[2]['batch_size']}\t{i[2]['conv_channels1']}\t{i[2]['conv_channels2']}\t{i[2]['conv_radius1']}\t{i[2]['conv_radius2']}\t{i[2]['embedding_size']}\t{i[2]['entropy_regularization']}\t{i[2]['fc_units']}\t{i[2]['learning_rate']}\t{i[2]['lstm_units']}\t{i[2]['num_fc_layers']}\t{i[2]['num_lstm_layers']}\t{i[2]['predict_pairs']}\t{i[2]['reward_exponent']}\t{i[2]['reward_function']}\t{i[2]['state_radius_relative']}")

    # print(all_solved)
    # print('\n')
    # print('Incumbent:')
    # print(inc_config)
    # print('It achieved accuracies of %f (validation) and %f (test).'%(1-inc_loss, inc_test_loss))
    # worker = LearnaWorker('', 1, [], run_id='analyse')
    # cs = worker.get_configspace()
    # fanova = result.get_fANOVA_data(cs)
    # f = fANOVA(fanova)

    # print(fanova)

    # worker = LearnaWorker('', 1, [], run_id='analyse')
    # cs = worker.get_configspace()
    # # r = result.get_fANOVA_data(cs)
    # print('generate fanova data')
    # a, b, _ = result.get_fANOVA_data(cs)
    # b = np.array([np.float64(x) for x in b])
    # print('create fanova object')
    # f = fANOVA(a, b, cs)

    # # getting the 10 most important pairwise marginals sorted by importance
    # # best_margs = f.get_most_important_pairwise_marginals(n=10)
    # # print(best_margs)
    # print('create visualizer')
    # path = Path('results', 'fanova_test')
    # path.mkdir(parents=True, exist_ok=True)
    # vis = fanova.visualizer.Visualizer(f, cs, directory=path)
    # # # creating the plot of pairwise marginal:
    # # vis.plot_pairwise_marginal((0,2), resolution=20)
    # # creating all plots in the directory
    # print('generate plots')
    # # vis.create_most_important_pairwise_marginal_plots(n=2)
    # vis.plot_marginal(1)
    # vis.plot_marginal(2)
    # vis.plot_marginal(3)
    # vis.plot_marginal(4)
    # vis.plot_marginal(5)
    # vis.plot_marginal(6)
    # vis.plot_marginal(7)
    # vis.plot_marginal(8)
    # vis.plot_marginal(9)
    # vis.plot_marginal(10)
    # vis.plot_marginal(11)
    # vis.plot_marginal(12)
    # vis.plot_marginal(13)
    # vis.plot_marginal(14)
    # vis.plot_marginal(15)
    # vis.plot_marginal(16)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run", type=str, help="The id of the run"
    )

    args = parser.parse_args()

    analyse_bohb_run(args.run)
