import time

import tensorflow as tf
from ..tensorforce.runner import Runner
from RNA import fold

from .agent import NetworkConfig, get_network, AgentConfig, get_agent_fn
from .environment import RnaDesignEnvironment, RnaDesignEnvironmentConfig


def _get_episode_finished(timeout, stop_once_solved):
    """
    Check for timeout after each episode of designing one entire target structure.

    Args:
        timeout: Maximum time allowed to solve one target structure.
        stop_once_solved: Defines if agent should stop after solving a target structure.

    Returns:
        episode_finish: Inner function that handles timeout.
    """
    start_time = time.time()

    def episode_finished(runner):
        env = runner.environment

        candidate_solution = env.design.primary
        last_reward = runner.episode_rewards[-1]
        # folding = fold(candidate_solution)[0]
        # last_fractional_hamming = env.episodes_info[-1].normalized_hamming_distance
        # last_gc_content = env.episodes_info[-1].gc_content
        # agent_gc = env.episodes_info[-1].agent_gc
        # gc_satisfied = env.episodes_info[-1].gc_satisfied
        elapsed_time = time.time() - start_time
        # print(elapsed_time, last_reward, last_fractional_hamming, gc_satisfied, last_gc_content, agent_gc, candidate_solution)
        print(elapsed_time, last_reward, candidate_solution)

        no_timeout = not timeout or elapsed_time < timeout
        stop_since_solved = stop_once_solved and last_reward == 1.0
        keep_running = not stop_since_solved and no_timeout
        return keep_running

    return episode_finished


def design_rna(
    dot_brackets,
    timeout,
    restore_path,
    stop_learning,
    restart_timeout,
    network_config,
    agent_config,
    env_config,
):
    """
    Main function for RNA design. Instantiate an environment and an agent to run in a
    tensorforce runner.

    Args:
        TODO
        timeout: Maximum time to run.
        restore_path: Path to restore saved configurations/models from.
        stop_learning: If set, no weight updates are performed (Meta-LEARNA).
        restart_timeout: Time interval for restarting of the agent.
        network_config: The configuration of the network.
        agent_config: The configuration of the agent.
        env_config: The configuration of the environment.

    Returns:
        Episode information.
    """
    session_config = tf.ConfigProto(
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1,
        allow_soft_placement=True,
        device_count={"CPU": 1},
    )

    env_config.use_conv = any(map(lambda x: x > 1, network_config.conv_sizes))
    env_config.use_embedding = bool(network_config.embedding_size)
    environment = RnaDesignEnvironment(dot_brackets, env_config)

    network = get_network(network_config)
    # Runner restarts the agent by calling get_agent again
    get_agent = get_agent_fn(
        environment=environment,
        network=network,
        agent_config=agent_config,
        session_config=session_config,
        restore_path=restore_path,
    )
    runner = Runner(get_agent, environment)

    stop_once_solved = len(dot_brackets) == 1
    runner.run(
        deterministic=False,
        restart_timeout=restart_timeout,
        stop_learning=stop_learning,
        episode_finished=_get_episode_finished(timeout, stop_once_solved),
    )
    return environment.episodes_info


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    from ..data.parse_dot_brackets import parse_dot_brackets, parse_local_design_data

    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument(
        "--target_structure_path", default=None, type=Path, help="Path to sequence to run on"
    )
    parser.add_argument("--data_dir", default="data", help="Data directory")
    parser.add_argument("--dataset", type=Path, help="Available: eterna, rfam_taneda")
    parser.add_argument(
        "--target_structure_ids",
        default=None,
        required=False,
        type=int,
        nargs="+",
        help="List of target structure ids to run on",
    )

    # Model
    parser.add_argument("--restore_path", type=Path, help="From where to load model")
    parser.add_argument("--stop_learning", action="store_true", help="Stop learning")
    parser.add_argument("--agent", type=str, help="Select the agent, available choices:trpo, ppo, random")  # TRPO doesn't work, error in optimizers/solvers/conugate_gradient.py IndexedSlice does not have attribute get_shape

    # Timeout behaviour
    parser.add_argument("--timeout", default=None, type=int, help="Maximum time to run")

    # Hyperparameters
    parser.add_argument("--learning_rate", type=float, help="Learning rate to use")
    parser.add_argument(
        "--mutation_threshold", type=int, help="Enable MUTATION with set threshold"
    )
    parser.add_argument(
        "--reward_exponent", default=1, type=float, help="Exponent for reward shaping"
    )
    parser.add_argument(
        "--state_radius", default=0, type=int, help="Radius around current site"
    )
    parser.add_argument(
        "--conv_sizes", type=int, default=[1], nargs="+", help="Size of conv kernels"
    )
    parser.add_argument(
        "--conv_channels", type=int, default=[50], nargs="+", help="Channel size of conv"
    )
    parser.add_argument(
        "--num_fc_layers", type=int, default=2, help="Number of FC layers to use"
    )
    parser.add_argument(
        "--fc_units", type=int, default=50, help="Number of units to use per FC layer"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for ppo agent"
    )
    parser.add_argument(
        "--entropy_regularization", type=float, default=1.5e-3, help="The output entropy"
    )
    parser.add_argument(
        "--restart_timeout", type=int, help="Time after which to restart the agent"
    )
    parser.add_argument("--lstm_units", type=int, help="The number of lstm units")
    parser.add_argument("--num_lstm_layers", type=int, help="The number of lstm layers")
    parser.add_argument("--embedding_size", type=int, help="The size of the embedding")
    # parser.add_argument("--gc_tolerance", default=0.04, type=float, help="The tolerance of the gc-content")
    # parser.add_argument("--desired_gc", default=0.5, type=float, help="The desired gc-content of the solution")
    # parser.add_argument("--gc_improvement_step", action="store_true", help="Control the gc-content of the solution")
    # parser.add_argument("--gc_reward", action="store_true", help="Include gc-content into reward function")
    # parser.add_argument("--gc_weight", default=1.0, type=float, help="The weighting factor for the gc-content constraint")
    # parser.add_argument("--num_actions", default=4, type=int, help="The number of actions that the agent chooses from")
    # parser.add_argument("--keep_sequence", default="fully", type=str, help="How much of the sequence of targets for local design is kept: fully, partially, no")
    parser.add_argument("--reward_function", type=str, default='structure_only', help="Decide if hamming distance is computed based on the folding only or also on the sequence parts")
    # parser.add_argument("--training_data", default="random", type=str, help="Choose the training data for local design: random sequences, motif based sequences")
    parser.add_argument("--local_design", action="store_true", help="Choose if agent should do RNA local Design")
    parser.add_argument("--predict_pairs", action="store_true", help="Choose if Actions are used to directly predict watson-crick base pairs")
    parser.add_argument("--state_representation", type=str, default='n-gram', help="Choose between n-gram and sequence_progress to show the nucleotides already placed in the state")
    parser.add_argument("--data_type", type=str, default='random', help="Choose type of training data, random motifs or motifs with balanced brackets")
    parser.add_argument("--sequence_constraints", type=str, default='-', help="Perform local design with knowledge about sequence")

    # parser.add_argument("--structure_only", action="store_true", help="Choose if state only considers structure parts of the target")


    args = parser.parse_args()

    print(f"args: \n {args}")

    network_config = NetworkConfig(
        conv_sizes=args.conv_sizes,  # radius * 2 + 1
        conv_channels=args.conv_channels,
        num_fc_layers=args.num_fc_layers,
        fc_units=args.fc_units,
        lstm_units=args.lstm_units,
        num_lstm_layers=args.num_lstm_layers,
        embedding_size=args.embedding_size,
    )
    agent_config = AgentConfig(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        entropy_regularization=args.entropy_regularization,
        agent=args.agent,
    )
    env_config = RnaDesignEnvironmentConfig(
        mutation_threshold=args.mutation_threshold,
        reward_exponent=args.reward_exponent,
        state_radius=args.state_radius,
        # gc_tolerance=args.gc_tolerance,
        # desired_gc=args.desired_gc,
        # gc_improvement_step=args.gc_improvement_step,
        # gc_weight=args.gc_weight,
        # gc_reward=args.gc_reward,
        local_design=args.local_design,
        # num_actions=args.num_actions,
        # keep_sequence=args.keep_sequence,
        # sequence_reward=args.sequence_reward,
        reward_function=args.reward_function,
        predict_pairs=args.predict_pairs,
        state_representation=args.state_representation,
        data_type=args.data_type,
        sequence_constraints=args.sequence_constraints,
        # structure_only=args.structure_only,
        # training_data=args.training_data,
    )
    dot_brackets = parse_dot_brackets(
        dataset=args.dataset,
        data_dir=args.data_dir,
        target_structure_ids=args.target_structure_ids,
        target_structure_path=args.target_structure_path,
    )
    if args.local_design:
        dot_brackets = parse_local_design_data(
            dataset=args.dataset,
            data_dir=args.data_dir,
            target_structure_ids=args.target_structure_ids,
            target_structure_path=args.target_structure_path,
        )
    # print(dot_brackets)

    design_rna(
        dot_brackets,
        timeout=args.timeout,
        restore_path=args.restore_path,
        stop_learning=args.stop_learning,
        restart_timeout=args.restart_timeout,
        network_config=network_config,
        agent_config=agent_config,
        env_config=env_config,
    )
