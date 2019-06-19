import time

from itertools import product
from dataclasses import dataclass
from distance import hamming

import numpy as np
from tensorforce.environments import Environment

from RNA import fold


@dataclass
class RnaDesignEnvironmentConfig:
    """
    Dataclass for the configuration of the environment.

    Default values describe:
        mutation_threshold: Defines the minimum distance needed before applying the local
            improvement step.
        reward_exponent: A parameter to shape the reward function.
        state_radius: The state representation is a (2*<state_radius> + 1)-gram
            at each position.
        use_conv: Bool to state if a convolutional network is used or not.
        use_embedding: Bool to state if embedding is used or not.
    """

    mutation_threshold: int = 5
    reward_exponent: float = 1.0
    state_radius: int = 5
    use_conv: bool = True
    use_embedding: bool = False
    gc_improvement_step: bool = False
    gc_postprocessing: bool = False
    gc_tolerance: float = 0.04
    desired_gc: float = 0.5
    gc_weight: float = 1.0
    structural_weight: float = 1.0
    gc_reward: bool = False


def _string_difference_indices(s1, s2):
    """
    Returns all indices where s1 and s2 differ.

    Args:
        s1: The first sequence.
        s2: The second sequence.

    Returns:
        List of indices where s1 and s2 differ.
    """
    return [index for index in range(len(s1)) if s1[index] != s2[index]]


def _encode_dot_bracket(secondary, env_config):
    """
    Encode the dot_bracket notated target structure. The encoding can either be binary
    or by the embedding layer.

    Args:
        secondary: The target structure in dot_bracket notation.
        env_config: The configuration of the environment.

    Returns:
        List of encoding for each site of the padded target structure.
    """
    padding = "=" * env_config.state_radius
    padded_secondary = padding + secondary + padding

    if env_config.use_embedding:
        site_encoding = {".": 0, "(": 1, ")": 2, "=": 3}
    else:
        site_encoding = {".": 0, "(": 1, ")": 1, "=": 0}

    # Sites corresponds to 1 pixel with 1 channel if convs are applied directly
    if env_config.use_conv and not env_config.use_embedding:
        return [[site_encoding[site]] for site in padded_secondary]
    return [site_encoding[site] for site in padded_secondary]


def _encode_pairing(secondary):
    """TODO
    """
    pairing_encoding = [None] * len(secondary)
    stack = []
    for index, symbol in enumerate(secondary, 0):
        if symbol == "(":
            stack.append(index)
        elif symbol == ")":
            paired_site = stack.pop()
            pairing_encoding[paired_site] = index
            pairing_encoding[index] = paired_site
    return pairing_encoding


class ConstraintControler(object):
    """TODO
    Class that provides error signals for all different types of constraints, e.g. structural constraints (via hamming distance), gc-content constraints, etc.
    """

    def __init__(self, gc_tolerance, desired_gc):
        self._gc_tolerance = gc_tolerance
        self._desired_gc = desired_gc

    def hamming_distance(self, folded_design, target):
        return hamming(folded_design, target.dot_bracket)

    def normalized_hamming_distance(self, folded_design, target):
        return self.hamming_distance(folded_design, target) / len(target)

    def gc_content(self, design):
        return (design.primary.upper().count('G') + design.primary.upper().count('C')) / len(design.primary)

    def gc_diff(self, design):
        return self.gc_content(design) - self.desired_gc

    def gc_diff_abs(self, design):
        return abs(self.gc_diff(design))

    def gc_satisfied(self, design):
        return self.gc_diff_abs(design) <= self._gc_tolerance

    @property
    def gc_tolerance(self):
        return self._gc_tolerance

    @property
    def desired_gc(self):
        return self._desired_gc


class LocalImprovement(object):
    """TODO
    Class for local improvement of the designed sequence. Improvement could be structure-based or gc-based.
    """

    def __init__(self, design, target, constraint_controller):
        self._design = design
        self._target = target
        self._constraint_controller = constraint_controller

    def structural_improvement_step(self):
        folded_design = fold(self._design.primary)[0]

        differing_sites = _string_difference_indices(
            self._target.dot_bracket, folded_design
        )

        hamming_distances = []
        for mutation in product("AGCU", repeat=len(differing_sites)):
            mutated = self._design.get_mutated(mutation, differing_sites)
            hamming_distance = self._constraint_controller.hamming_distance(fold(mutated.primary)[0], self._target)
            hamming_distances.append((mutated, hamming_distance))
            if hamming_distance == 0:  # For better timing results
                return mutated, 0
        return min(hamming_distances, key = lambda t: t[1])

    def gc_improvement_step(self, hamming_distance=0, site=0):
        single_sites = []

        if self._constraint_controller.gc_satisfied(self._design):
            print("Entered gc_improvement gc_satisfied")
            return self._design

        # Start improvement by replacing nucleotides at paired sites
        while not self._constraint_controller.gc_satisfied(self._design):
            # print(f"improved design at {site}: {self._design.primary}")
            # stop if end of designed primary is reached
            if site >= len(self._design.primary):
                break

            # single sites are stored for later use
            if not self._target.get_paired_site(site):
                single_sites.append(site)
                site += 1
                continue

            # Decrease or increase gc-content
            if self._constraint_controller.gc_diff(self._design) < 0:
                self._increase_gc(site, self._target.get_paired_site(site), hamming_distance)
            else:
                self._decrease_gc(site, self._target.get_paired_site(site), hamming_distance)

            site += 1

        # if gc constraint is not satisfied, iterate the single sites and improve gc-content
        if not self._constraint_controller.gc_satisfied(self._design):
            print("Starting single site improvement")
            return self._gc_improve_single_sites(single_sites, hamming_distance=hamming_distance)

        print("Success via paired sites!!")
        return self._design

    def _gc_improve_single_sites(self, single_sites, hamming_distance=0):
        for site in single_sites:
            # print(f"Single improved design at {site}: {self._design.primary}")
            if not self._constraint_controller.gc_satisfied(self._design):
                if self._constraint_controller.gc_diff(self._design) < 0:
                    self._increase_gc(site, self._target.get_paired_site(site), hamming_distance)
                else:
                    self._decrease_gc(site, self._target.get_paired_site(site), hamming_distance)
            else:
                print("gc_satisfied via single sites")
                return self._design
        print("gc NOT satisfied via single sites")
        return self._design

    def _increase_gc(self, site, paired_site=None, hamming_distance=0):
        primary = list(self._design.primary)
        if self._design.primary[site] == 'A':
            primary[site] = np.random.choice(['G', 'C'])  # currently random choice, could be improved
            if paired_site:
                primary[site] = 'G'
                primary[paired_site] = 'C'
            if hamming(fold(''.join(primary))[0], self._target.dot_bracket) == hamming_distance:
                self._design.assign_sites(0, site, self._target.get_paired_site(site))
                print(f"Changed nucleotide {site}")
            else:
                return
        elif self._design.primary[site] == 'U':
            primary[site] = np.random.choice(['G', 'C'])  # currently rnadom choice, could be improved
            if paired_site:
                primary[site] = 'C'
                primary[paired_site] = 'G'
            if hamming(fold(''.join(primary))[0], self._target.dot_bracket) == hamming_distance:
                    self._design.assign_sites(1, site, self._target.get_paired_site(site))
                    print(f"Changed nucleotide {site}")
            else:
                return

    # not implemented for this class!!!
    def _decrease_gc(self, site, paired_site=None, hamming_distance=0):
        primary = list(self._design.primary)
        if self._design.primary[site] == 'G':
            primary[site] = np.random.choice(['A', 'U'])
            if paired_site:
                primary[site] = 'A'
                primary[paired_site] = 'U'
            if hamming(fold(''.join(primary))[0], self._target.dot_bracket) == hamming_distance:
                self._design.assign_sites(2, site, self._target.get_paired_site(site))
                print(f"Changed nucleotide {site}")
            else:
                return
        elif self._design.primary[site] == 'C':
            primary[site] = np.random.choice(['A', 'U'])
            if paired_site:
                primary[site] = 'U'
                primary[paired_site] = 'A'
            if hamming(fold(''.join(primary))[0], self._target.dot_bracket) == hamming_distance:
                    self._design.assign_sites(3, site, self._target.get_paired_site(site))
                    print(f"Changed nucleotide {site}")
            else:
                return

class _Target(object):
    """TODO
    Class of the target structure. Provides encodings and id.
    """

    _id_counter = 0

    def __init__(self, dot_bracket, env_config):
        """
        Initialize a target structure.

        Args:
             dot_bracket: dot_bracket encoded target structure.
             env_config: The environment configuration.
        """
        _Target._id_counter += 1
        self.id = _Target._id_counter  # For processing results
        self.dot_bracket = dot_bracket
        self._pairing_encoding = _encode_pairing(self.dot_bracket)
        self.padded_encoding = _encode_dot_bracket(self.dot_bracket, env_config)

    def __len__(self):
        return len(self.dot_bracket)

    def get_paired_site(self, site):
        """
        Get the paired site for <site> (base pair).

        Args:
            site: The site to check the pairing site for.

        Returns:
            The site that pairs with <site> if exists.TODO
        """
        return self._pairing_encoding[site]


class _Design(object):
    """
    Class of the designed candidate solution.
    """

    action_to_base = {0: "G", 1: "A", 2: "U", 3: "C"}
    action_to_pair = {0: "GC", 1: "CG", 2: "AU", 3: "UA"}

    def __init__(self, length=None, primary=None):
        """
        Initialize a candidate solution.

        Args:
            length: The length of the candidate solution.
            primary: The sequence of the candidate solution.
        """
        if primary:
            self._primary_list = primary
        else:
            self._primary_list = [None] * length
        self._dot_bracket = None
        self._current_site = 0

    def get_mutated(self, mutations, sites):
        """
        Locally change the candidate solution.

        Args:
            mutations: Possible mutations for the specified sites
            sites: The sites to be mutated

        Returns:
            A Design object with the mutated candidate solution.
        """
        mutatedprimary = self._primary_list.copy()
        for site, mutation in zip(sites, mutations):
            mutatedprimary[site] = mutation
        return _Design(primary=mutatedprimary)

    def assign_sites(self, action, site, paired_site=None):
        """
        Assign nucleotides to sites for designing a candidate solution.

        Args:
            action: The agents action to assign a nucleotide.
            site: The site to which the nucleotide is assigned to.
            paired_site: defines if the site is assigned with a base pair or not.
        """
        self._current_site += 1
        if paired_site:
            base_current, base_paired = self.action_to_pair[action]
            self._primary_list[site] = base_current
            self._primary_list[paired_site] = base_paired
        else:
            self._primary_list[site] = self.action_to_base[action]

    @property
    def first_unassigned_site(self):
        try:
            while self._primary_list[self._current_site] is not None:
                self._current_site += 1
            return self._current_site
        except IndexError:
            return None

    @property
    def primary(self):
        return "".join(self._primary_list)


def _random_epoch_gen(data):
    """
    Generator to get epoch data.

    Args:
        data: The targets of the epoch
    """
    while True:
        for i in np.random.permutation(len(data)):
            yield data[i]


@dataclass
class EpisodeInfo:
    """
    Information class.
    """

    __slots__ = ["target_id", "time", "normalized_hamming_distance", "gc_content", "agent_gc", "delta_gc", "gc_satisfied"]
    target_id: int
    time: float
    normalized_hamming_distance: float
    gc_content: float
    agent_gc: float
    delta_gc: float
    gc_satisfied: bool




class RnaDesignEnvironment(Environment):
    """
    The environment for RNA design using deep reinforcement learning.
    """

    def __init__(self, dot_brackets, env_config):
        """TODO
        Initialize an environemnt.
gc control
        Args:
            env_config: The configuration of the environment.
        """
        self._env_config = env_config

        targets = [_Target(dot_bracket, self._env_config) for dot_bracket in dot_brackets]
        self._target_gen = _random_epoch_gen(targets)

        self.target = None
        self.design = None
        # print(self._env_config.gc_tolerance, self._env_config.desired_gc)
        self._constraint_controller = ConstraintControler(self._env_config.gc_tolerance, self._env_config.desired_gc)
        self.episodes_info = []

    def __str__(self):
        return "RnaDesignEnvironment"

    def seed(self, seed):
        return None

    def reset(self):
        """
        Reset the environment. First function called by runner. Returns first state.

        Returns:
            The first state.
        """
        self.target = next(self._target_gen)
        self.design = _Design(len(self.target))
        return self._get_state()

    def _apply_action(self, action):
        """
        Assign a nucleotide to a site.

        Args:
            action: The action chosen by the agent.
        """
        current_site = self.design.first_unassigned_site
        paired_site = self.target.get_paired_site(current_site)  # None for unpaired sites
        self.design.assign_sites(action, current_site, paired_site)

    def _get_state(self):
        """
        Get a state dependend on the padded encoding of the target structure.

        Returns:
            The next state.
        """
        start = self.design.first_unassigned_site
        return self.target.padded_encoding[
            start : start + 2 * self._env_config.state_radius + 1
        ]

    def _get_reward(self, terminal):
        """
        Compute the reward after assignment of all nucleotides.

        Args:
            terminal: Bool defining if final timestep is reached yet.

        Returns:
            The reward at the terminal timestep or 0 if not at the terminal timestep.
        """
        if not terminal:
            return 0

        agent_gc = self._constraint_controller.gc_diff(self.design)

        hamming_distance = self._constraint_controller.hamming_distance(fold(self.design.primary)[0], self.target)

        # start local improvement procedure
        _local_improvement = LocalImprovement(self.design, self.target, self._constraint_controller)
        if 0 < hamming_distance < self._env_config.mutation_threshold:
            # improve for structural constraint
            self.design, hamming_distance = _local_improvement.structural_improvement_step()
            # improve for gc-content constraint
            if self._env_config.gc_improvement_step:
                self.design = _local_improvement.gc_improvement_step(hamming_distance=hamming_distance)
                print(f"(GIS) GC before: {agent_gc} GC after: {self._constraint_controller.gc_diff(self.design)}")

        # apply gc_control as postprocessing step if hamming distance is 0
        if hamming_distance == 0 and self._env_config.gc_postprocessing:
            self.design = _local_improvement.gc_improvement_step(hamming_distance=hamming_distance)

        normalized_hamming_distance = self._constraint_controller.normalized_hamming_distance(fold(self.design.primary)[0], self.target)
        delta_gc = self._constraint_controller.gc_diff_abs(self.design)

        # For hparam optimization
        episode_info = EpisodeInfo(
            target_id=self.target.id,
            time=time.time(),
            normalized_hamming_distance=normalized_hamming_distance,
            gc_content=self._constraint_controller.gc_content(self.design),
            agent_gc=agent_gc,
            delta_gc=delta_gc,
            gc_satisfied=self._constraint_controller.gc_satisfied(self.design)
        )
        self.episodes_info.append(episode_info)


        if hamming_distance == 0 and self._constraint_controller.gc_satisfied(self.design):
            return 1.0
        # Jointly optimize for gc-content and structural constraint
        if self._env_config.gc_reward:
            return (1 - (self._env_config.structural_weight * normalized_hamming_distance) - (self._env_config.gc_weight * delta_gc)) ** self._env_config.reward_exponent

        # TODO(): Utilize bool or something to handle stopping only if gc is chosen.
        if hamming_distance == 0 and not self._constraint_controller.gc_satisfied(self.design):
            print("Entered hamming == 0 and not gc_satisfied!!")
            return (1 - self._env_config.gc_tolerance) ** self._env_config.reward_exponent

        # Else return normalized hamming distance
        return (1 - normalized_hamming_distance) ** self._env_config.reward_exponent

    def execute(self, actions):
        """
        Execute one interaction of the environment with the agent.

        Args:
            action: Current action of the agent.

        Returns:
            state: The next state for the agent.
            terminal: The signal for end of an episode.
            reward: The reward if at terminal timestep, else 0.
        """
        self._apply_action(actions)

        terminal = self.design.first_unassigned_site is None
        state = None if terminal else self._get_state()
        reward = self._get_reward(terminal)

        return state, terminal, reward

    def close(self):
        pass

    @property
    def states(self):
        type = "int" if self._env_config.use_embedding else "float"
        if self._env_config.use_conv and not self._env_config.use_embedding:
            return dict(type=type, shape=(1 + 2 * self._env_config.state_radius, 1))
        return dict(type=type, shape=(1 + 2 * self._env_config.state_radius,))

    @property
    def actions(self):
        return dict(type="int", num_actions=4)
