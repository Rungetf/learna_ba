import ConfigSpace as CS
from ConfigSpace.read_and_write import json
from pathlib import Path

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
            "embedding_size", lower=0, upper=8, default_value=1  # FR: changed embedding size upper from 4 to 8
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
            "embedding_size", lower=0, upper=8, default_value=1  # FR: changed embedding size upper from 4 to 8
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
            "embedding_size", lower=0, upper=8, default_value=1  # FR: changed embedding size upper from 4 to 8
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


if __name__ ==  '__main__':
    output_dir = 'results/bohb/6826895/'
    config_space = get_fine_tuning_config()
    print(config_space)
    out_file = Path(output_dir, 'configspace.pcs')

    with open(out_file, 'w') as fh:
        fh.write(json.write(config_space))
