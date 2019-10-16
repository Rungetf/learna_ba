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
import pandas as pd


def read_data(path, run):
    return hpres.logged_results_to_HBS_result(Path(path, run))


def reward_vs_loss(result, path, run, out_dir):
    data = result.get_pandas_dataframe()

    reward_functions = [(reward, loss) for reward, loss in zip(data[0].reward_function, data[1].loss)]
    reward_df = pd.DataFrame(reward_functions, columns=['reward_function', 'loss'])

    structure_only_df = reward_df.where(lambda x: x.reward_function == 'structure_only').dropna()
    structure_only_df['reward_function'] = structure_only_df.index

    sequence_and_structure_df = reward_df.where(lambda x: x.reward_function == 'sequence_and_structure').dropna()
    sequence_and_structure_df['reward_function'] = sequence_and_structure_df.index

    structure_replace_sequence_df = reward_df.where(lambda x: x.reward_function == 'structure_replace_sequence').dropna()
    structure_replace_sequence_df['reward_function'] = structure_replace_sequence_df.index

    path = Path(out_dir, run, 'reward_function')
    path.mkdir(exist_ok=True, parents=True)

    structure_only_df.to_csv(path_or_buf=Path(path, 'structure_only.tsv'), sep='\t', index=False)
    sequence_and_structure_df.to_csv(path_or_buf=Path(path, 'sequence_and_structure.tsv'), sep='\t', index=False)
    structure_replace_sequence_df.to_csv(path_or_buf=Path(path, 'structure_replace_sequence.tsv'), sep='\t', index=False)


def action_vs_loss(result, path, run, out_dir):
    data = result.get_pandas_dataframe()

    action_semantics = [(prediction, loss) for prediction, loss in zip(data[0].predict_pairs, data[1].loss)]
    action_df = pd.DataFrame(action_semantics, columns=['prediction', 'loss'])

    pair_prediction_df = action_df.where(lambda x: x.prediction == 1).dropna()
    pair_prediction_df['prediction'] = pair_prediction_df.index

    no_pair_prediction_df = action_df.where(lambda x: x.prediction == 0).dropna()
    no_pair_prediction_df['prediction'] = no_pair_prediction_df.index

    path = Path(out_dir, run, 'pair_prediction')
    path.mkdir(exist_ok=True, parents=True)

    pair_prediction_df.to_csv(path_or_buf=Path(path, 'pair_prediction.tsv'), sep='\t', index=False)
    no_pair_prediction_df.to_csv(path_or_buf=Path(path, 'no_pair_prediction.tsv'), sep='\t', index=False)


def budgets_vs_loss(result, path, run, out_dir):
    data = result.get_pandas_dataframe()

    budgets = [(budget, loss) for budget, loss in zip(data[0].budget, data[1].loss)]
    budgets_df = pd.DataFrame(budgets, columns=['budget', 'loss'])

    budgets = set(budgets_df['budget'].values)

    path = Path(out_dir, run, 'budgets')
    path.mkdir(exist_ok=True, parents=True)

    for budget in budgets:
        budget_df = budgets_df.where(lambda x: x.budget == budget).dropna()
        budget_df['budget'] = budget_df.index
        budget_df.to_csv(path_or_buf=Path(path, f"{budget}.tsv"), sep='\t', index=False)


def nas_vs_loss(result, path, run, out_dir):
    data = result.get_pandas_dataframe()

    nas = [(fc, lstm, cnn1, cnn2, loss) for fc, lstm, cnn1, cnn2, loss in zip(data[0].num_fc_layers, data[0].num_lstm_layers, data[0].conv_radius1, data[0].conv_radius2, data[1].loss)]
    nas_df = pd.DataFrame(nas, columns=['fc', 'lstm', 'cnn1', 'cnn2', 'loss'])

    fc_only_df = nas_df.where(lambda x: x.lstm == 0).dropna()
    fc_only_df = fc_only_df.where(lambda x: x.cnn1 +x.cnn2 == 0).dropna()
    fc_only_df = fc_only_df[['fc', 'loss']]
    fc_only_df['fc'] = fc_only_df.index

    lstm_and_fc_df = nas_df.where(lambda x: x.lstm != 0).dropna()
    lstm_and_fc_df = lstm_and_fc_df.where(lambda x: x.cnn1 +x.cnn2 == 0).dropna()
    lstm_and_fc_df = lstm_and_fc_df[['lstm', 'loss']]
    lstm_and_fc_df['lstm'] = lstm_and_fc_df.index

    cnn_and_fc_df = nas_df.where(lambda x: x.cnn1 + x.cnn2 != 0).dropna()
    cnn_and_fc_df = cnn_and_fc_df.where(lambda x: x.lstm == 0).dropna()
    cnn_and_fc_df = cnn_and_fc_df[['cnn1', 'loss']]
    cnn_and_fc_df['cnn1'] = cnn_and_fc_df.index

    lstm_cnn_fc_df = nas_df.where(lambda x: x.cnn1 + x.cnn2 != 0).dropna()
    lstm_cnn_fc_df = lstm_cnn_fc_df.where(lambda x: x.lstm != 0).dropna()
    lstm_cnn_fc_df = lstm_cnn_fc_df[['fc', 'loss']]
    lstm_cnn_fc_df['fc'] = lstm_cnn_fc_df.index

    path = Path(out_dir, run, 'NAS')
    path.mkdir(exist_ok=True, parents=True)

    fc_only_df.to_csv(path_or_buf=Path(path, 'fc.tsv'), sep='\t', index=False)
    lstm_and_fc_df.to_csv(path_or_buf=Path(path, 'lstm.tsv'), sep='\t', index=False)
    cnn_and_fc_df.to_csv(path_or_buf=Path(path, 'cnn.tsv'), sep='\t', index=False)
    lstm_cnn_fc_df.to_csv(path_or_buf=Path(path, 'lstm_cnn_fc.tsv'), sep='\t', index=False)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--path", type=str, help="results directory"
    )

    parser.add_argument(
        "--run", type=str, help="The run id"
    )

    parser.add_argument(
        "--out_dir", type=str, help="The output directory"
    )

    args = parser.parse_args()

    print('Read data')
    result = read_data(args.path, args.run)

    print('Reward vs loss')
    reward_vs_loss(result, args.path, args.run, args.out_dir)

    print('Actions vs loss')
    action_vs_loss(result, args.path, args.run, args.out_dir)

    print('Budets vs loss')
    budgets_vs_loss(result, args.path, args.run, args.out_dir)

    print('NAS vs loss')
    nas_vs_loss(result, args.path, args.run, args.out_dir)
