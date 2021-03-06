import time
import re

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
        gc_improvement_step: Bool to decide if gc_improvement step is used.
        gc_tolerance: The tolerance for gc content.
        desired_gc: The gc_content that is desired.
        gc_weight: Determines how much weight is set on the gc content control.
        gc_reward: Bool to decide if the gc_content is included in the reward function.
        num_actions: The number of actions the agent can choose from.
    """

    mutation_threshold: int = 5
    reward_exponent: float = 1.0
    state_radius: int = 5
    use_conv: bool = True
    use_embedding: bool = False
    local_design: bool = False
    predict_pairs: bool = False
    reward_function: str = 'structure'
    state_representation: str = 'sequence-progress'
    data_type: str = 'random'
    sequence_constraints: str = None


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
        site_encoding = {".": 0, "(": 1, ")": 2, "=": 3, "A": 4, "G": 5, "C": 6, "U": 7, "N": 8}  # add acgu add N then everything works with every constraint formulation; change N: 8 if ready
    else:
        site_encoding = {".": 0, "(": 1, ")": 1, "=": 0, "A": 2, "G": 3, "C": 4, "U": 5, "N": 6}  # add acgu

    # Sites corresponds to 1 pixel with 1 channel if convs are applied directly
    if env_config.use_conv and not env_config.use_embedding:
        return [[site_encoding[site]] for site in padded_secondary]
    return [site_encoding[site] for site in padded_secondary]


def _encode_pairing(secondary):
    """
    Encode the base pairing scheme of the design task.

    Args:
        secondary: the secondary structure (domain) that has to get encoded.

    Returns:
        List that contains the paired site for each site if exists, or None otherwise.
    """
    pairing_encoding = [None] * len(secondary)
    stack = []
    for index, symbol in enumerate(secondary, 0):
        if symbol == "(":
            stack.append(index)
        elif symbol == ")":
            if not stack:
                continue
            else:
                paired_site = stack.pop()
                pairing_encoding[paired_site] = index
                pairing_encoding[index] = paired_site
    return pairing_encoding


def _encode_structure_parts(local_target):
    """
    Encodes the design task, to distinguish between struture and wequence domains.

    Args:
        A sequence that may contain structure or sequence parts.

    Returns:
        A list representation of the sequence with all structure sites and None else.
    """
    encoding = [None] * len(local_target)
    for index, site in enumerate(local_target):
        if site in ['.', '(', ')', 'N']:
            encoding[index] = site
    return encoding


#class ConstraintControler(object):
#    """TODO
#    Class that provides error signals for all different types of constraints, e.g. structural constraints (via hamming distance), gc-content constraints, etc.
#    """
#
#    def __init__(self, gc_tolerance, desired_gc):
#        self._gc_tolerance = gc_tolerance
#        self._desired_gc = desired_gc
#
#    def hamming_distance(self, folded_design, target):
#        return hamming(folded_design, target.dot_bracket)
#
#    def normalized_hamming_distance(self, folded_design, target):
#        return self.hamming_distance(folded_design, target) / len(target)
#
#    def gc_content(self, design):
#        return (design.primary.upper().count('G') + design.primary.upper().count('C')) / len(design.primary)
#
#    def gc_diff(self, design):
#        return self.gc_content(design) - self.desired_gc
#
#    def gc_diff_abs(self, design):
#        return abs(self.gc_diff(design))
#
#    def gc_satisfied(self, design):
#        return self.gc_diff_abs(design) <= self._gc_tolerance
#
#    @property
#    def gc_tolerance(self):
#        return self._gc_tolerance
#
#    @property
#    def desired_gc(self):
#        return self._desired_gc
#
#
#class LocalImprovement(object):
#    """TODO
#    Class for local improvement of the designed sequence. Improvement could be structure-based or gc-based.
#    """
#
#    def __init__(self, design, target, constraint_controller):
#        self._design = design
#        self._target = target
#        self._constraint_controller = constraint_controller
#
#    def structural_improvement_step(self):
#        folded_design = fold(self._design.primary)[0]
#
#        differing_sites = _string_difference_indices(
#            self._target.dot_bracket, folded_design
#        )
#
#        hamming_distances = []
#        for mutation in product("AGCU", repeat=len(differing_sites)):
#            mutated = self._design.get_mutated(mutation, differing_sites)
#            hamming_distance = self._constraint_controller.hamming_distance(fold(mutated.primary)[0], self._target)
#            hamming_distances.append((mutated, hamming_distance))
#            if hamming_distance == 0:  # For better timing results
#                return mutated, 0
#        return min(hamming_distances, key = lambda t: t[1])
#
#    def gc_improvement_step(self, hamming_distance=0, site=0):
#        single_sites = []
#
#        if self._constraint_controller.gc_satisfied(self._design):
#            return self._design
#
#        # Start improvement by replacing nucleotides at paired sites
#        while not self._constraint_controller.gc_satisfied(self._design):
#            # print(f"improved design at {site}: {self._design.primary}")
#            # stop if end of designed primary is reached
#            if site >= len(self._design.primary):
#                break
#
#            # single sites are stored for later use
#            if not self._target.get_paired_site(site):
#                single_sites.append(site)
#                site += 1
#                continue
#
#            # Decrease or increase gc-content
#            if self._constraint_controller.gc_diff(self._design) < 0:
#                self._increase_gc(site, self._target.get_paired_site(site), hamming_distance)
#            else:
#                self._decrease_gc(site, self._target.get_paired_site(site), hamming_distance)
#
#            site += 1
#
#        # if gc constraint is not satisfied, iterate the single sites and improve gc-content
#        if not self._constraint_controller.gc_satisfied(self._design):
#            return self._gc_improve_single_sites(single_sites, hamming_distance=hamming_distance)
#
#        return self._design
#
#    def _gc_improve_single_sites(self, single_sites, hamming_distance=0):
#        for site in single_sites:
#            # print(f"Single improved design at {site}: {self._design.primary}")
#            if not self._constraint_controller.gc_satisfied(self._design):
#                if self._constraint_controller.gc_diff(self._design) < 0:
#                    self._increase_gc(site, self._target.get_paired_site(site), hamming_distance)
#                else:
#                    self._decrease_gc(site, self._target.get_paired_site(site), hamming_distance)
#            else:
#                return self._design
#        return self._design
#
#    def _increase_gc(self, site, paired_site=None, hamming_distance=0):
#        primary = list(self._design.primary)
#        if self._design.primary[site] == 'A':
#            primary[site] = 'G'  # choice could be improved
#            if paired_site:
#                primary[site] = 'G'
#                primary[paired_site] = 'C'
#            if hamming(fold(''.join(primary))[0], self._target.dot_bracket) == hamming_distance:  # this should probably be <=
#                self._design.assign_sites(0, site, self._target.get_paired_site(site))  # TODO: predict_pairs is not in the call!!!!
#            else:
#                return
#        elif self._design.primary[site] == 'U':
#            primary[site] = 'C'  # choice could be improved
#            if paired_site:
#                primary[site] = 'C'
#                primary[paired_site] = 'G'
#            if hamming(fold(''.join(primary))[0], self._target.dot_bracket) == hamming_distance:  # this should probably be <=
#                    self._design.assign_sites(1, site, self._target.get_paired_site(site))  # TODO: predict_pairs is not in the call!!!!
#            else:
#                return
#
#    # not implemented for this class!!!
#    def _decrease_gc(self, site, paired_site=None, hamming_distance=0):
#        primary = list(self._design.primary)
#        if self._design.primary[site] == 'G':
#            primary[site] = 'A'
#            if paired_site:
#                primary[site] = 'A'
#                primary[paired_site] = 'U'
#            if hamming(fold(''.join(primary))[0], self._target.dot_bracket) == hamming_distance:  # this should probably be <=
#                self._design.assign_sites(2, site, self._target.get_paired_site(site))  # TODO: predict_pairs is not in the call!!!!
#            else:
#                return
#        elif self._design.primary[site] == 'C':
#            primary[site] = 'U'
#            if paired_site:
#                primary[site] = 'U'
#                primary[paired_site] = 'A'
#            if hamming(fold(''.join(primary))[0], self._target.dot_bracket) == hamming_distance:  # this should probably be <=
#                    self._design.assign_sites(3, site, self._target.get_paired_site(site))  # TODO: predict_pairs is not in the call!!!!
#            else:
#                return
#
class _Target(object):
    """TODO
    Class of the target structure. Provides encodings and id.
    """

    _id_counter = 0

    def __init__(self, dot_bracket, env_config):
        """
        Initialize a target structure.
        TODO: Make reading of the task more generic.

        Args:
             dot_bracket: the task information:
                 position 0: id
                 position 1: structure constraints
                 position 2: sequence constraints
                 position 3: gc content
                 position 4: mfe
             env_config: The environment configuration.
        """
        _Target._id_counter += 1
        self._env_config = env_config

        try:
            assert len(dot_bracket) == 5
            self.id = dot_bracket[0]
            self.dot_bracket = dot_bracket[1]
            self.sequence_constraints = dot_bracket[2] if dot_bracket[2] != '-' else None
            self.gc = dot_bracket[3]
            self.mfe = dot_bracket[4]
        except:
            try:
                self.id = dot_bracket[0]
                self.dot_bracket = dot_bracket[3]
                self.sequence = dot_bracket[2]
                self.local_motif = dot_bracket[4]
                self.gc = dot_bracket[5]
                self.mfe = dot_bracket[6]
                self.sequence_constraints = None
            except:
                 self.id = _Target._id_counter
                 self.dot_bracket = dot_bracket
                 self.sequence_constraints = None
                 self.gc = None
                 self.mfe = None

        if not self.sequence_constraints:
            sequence_constraints = ['N' for _ in self.dot_bracket]
            dot_bracket = [c for c in self.dot_bracket]
            for index, character in enumerate(self.dot_bracket):
                if character in ['A', 'C', 'G', 'U']:
                    sequence_constraints[index] = character
                    dot_bracket[index] = 'N'
            self.sequence_constraints = ''.join(sequence_constraints)
            self.dot_bracket = ''.join(dot_bracket)

        self._current_site = 0
        self._partition = self.get_partition()
        self.encode_partial_pairing()
        self.local_target = self.assign_sequence_constraints()

        self.sequence_progress = self.local_target
        self.padded_encoding = _encode_dot_bracket(self.local_target, env_config)

        if env_config.reward_function == 'structure_only':
            self.structure_parts_encoding = _encode_structure_parts(self.local_target)


    def encode_partial_pairing(self):
        """
        Calls _encode_pairing() for all structure parts of the structure constraints to get the pairing scheme correctly.
        """
        self._pairing_encoding = [None] * len(self.dot_bracket)

        for start, end in self.partition[1]:
            encoding = _encode_pairing(self.dot_bracket[start:end])
            for index, site in enumerate(encoding):
                if site is not None:
                    self._pairing_encoding[index + start] = site + start

    def assign_sequence_constraints(self):
        """
        Generate a joint task representation from the sequence and the structure constraints.
        """
        pair_assignments = {'A': 'U', 'U': 'A', 'G': 'C', 'C': 'G'}
        new_local_target = [site for site in self.dot_bracket.rstrip()]
        for index, site in enumerate(self.sequence_constraints):
            if site != 'N':
                paired_site = self._pairing_encoding[index]
                new_local_target[index] = site
                if paired_site:
                    new_local_target[paired_site] = pair_assignments[site]
        return ''.join(new_local_target)

    def get_partition(self):
        """
        Get the struture domains from the structure constraints and the sequence domains from the sequence constraints.
        """
        sequence_pattern = re.compile(r"[A, C, G, U]+")
        structure_pattern = re.compile(r"[0, 1, 2]+")
        sequence_parts = []
        structure_parts = []
        current_index = 0
        for pattern in sequence_pattern.findall(self.sequence_constraints):
            start, end = re.search(pattern, self.sequence_constraints[current_index:]).span()
            sequence_parts.append((start + current_index, end + current_index))
            current_index += end
        current_index = 0
        tmp_db = [x for x in self.dot_bracket.rstrip()]
        for index, site in enumerate(self.dot_bracket.rstrip()):
            if site not in ['N', 'A', 'C', 'G', 'U']:
                if site == '.':
                    tmp_db[index] = '0'
                elif site == '(':
                    tmp_db[index] = '1'
                else:
                    tmp_db[index] = '2'
        tmp_db = ''.join(tmp_db)
        for pattern in structure_pattern.findall(tmp_db):
            start, end = re.search(pattern, tmp_db[current_index:]).span()
            structure_parts.append((start + current_index, end + current_index))
            current_index += end
        return sequence_parts, structure_parts

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

    def reset(self):
        """
        A reset is required because the actual task representation is changed when the state is dependend on the previous action of the agent.
        """
        self.sequence_progress = self.local_target if self._env_config.local_design else self.dot_bracket
        self.padded_encoding = _encode_dot_bracket(self.dot_bracket, self._env_config) if not self._env_config.local_design else _encode_dot_bracket(self.local_target, self._env_config)

    def assign_sites(self, index, value, paired_site):
        """
        Assign values to sites of the task, to incorporate the agent's actions into the template for the state representations.

        Args:
            index: current site to assign a nucleotide to.
            value: the value to assign.
            paired_site: bool if site has a known pairing partner.
        """
        new_local_target = list(self.sequence_progress)
        if paired_site:
            new_local_target[index[0]] = value[0]
            new_local_target[index[1]] = value[1]
        else:
            new_local_target[index] = value
        self.sequence_progress = ''.join(new_local_target)
        self.padded_encoding = _encode_dot_bracket(self.sequence_progress, self._env_config)


    def reset_counter(self):
        """
        Resets the counter of the current site for predictions.
        """
        self._current_site = 0
        self.structure_parts_encoding = _encode_structure_parts(self.local_target)
        # print(f"target_reset: {self.structure_parts_encoding}")

    @property
    def partition(self):
        return self._partition

    @property
    def next_structure_site(self):
        try:
            while self.structure_parts_encoding[self._current_site] is None:
                self._current_site += 1
            self.structure_parts_encoding[self._current_site] = None
            if self._env_config.predict_pairs and self.get_paired_site(self._current_site):
                self.structure_parts_encoding[self.get_paired_site(self._current_site)] = None
            return self._current_site
        except IndexError:
            return None

    @property
    def current_site(self):
        return self._current_site


class _Design(object):
    """
    Class of the designed candidate solution.
    """

    action_to_base = {0: "G", 1: "A", 2: "U", 3: "C", 4:'_'}
    action_to_pair = {0: "GC", 1: "AU", 2: "UA", 3: "CG"}

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
        self._last_assignment = None

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

    def assign_sites(self, action, site, paired_site=None, predict_pairs=False):
        """
        Assign nucleotides to sites for designing a candidate solution.

        Args:
            action: The agents action to assign a nucleotide.
            site: The site to which the nucleotide is assigned to.
            paired_site: defines if the site is assigned with a base pair or not.
        """
        self._current_site += 1
        if predict_pairs and paired_site:
            base_current, base_paired = self.action_to_pair[action]
            self._primary_list[site] = base_current
            self._primary_list[paired_site] = base_paired
            self._last_assignment = ((site, paired_site), (base_current, base_paired), True)
        else:
            self._primary_list[site] = self.action_to_base[action]
            self._last_assignment = (site, self.action_to_base[action], False)

    @property
    def last_assignment(self):
        return self._last_assignment

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

def _sorted_data_gen(data):
    """
    Generator to get sorted epoch data
    """
    data = sorted(data, key=lambda x: len(x))
    while True:
        for target in data:
            yield target

def hamming_with_n(s1, s2):
    """
    Hamming Distance without counting wildcard smbols.

    Args:
        s1: the first sequence for comparison.
        s2: the second sequence for comparison.

    Returns:
        the distance without accounting unrestricted sites.
    """
    distance = 0
    for c1, c2 in zip(s1, s2):
        if c1 != c2:
            if c1 == 'N' or c2 == 'N':
                continue
            else:
                distance += 1
    return distance

@dataclass
class EpisodeInfo:
    """
    Information class.
    """

    __slots__ = ["target_id", "time", "normalized_hamming_distance", "gc_content", "agent_gc", "delta_gc", "gc_satisfied"]
    target_id: int
    time: float
    normalized_hamming_distance: float

class RnaDesignEnvironment(Environment):
    """
    The environment for RNA design using deep reinforcement learning.
    """

    def __init__(self, dot_brackets, env_config):
        """TODO
        Initialize an environemnt.

        Args:
            env_config: The configuration of the environment.
        """
        self._env_config = env_config

        targets = [_Target(dot_bracket, self._env_config) for dot_bracket in dot_brackets]
        self._target_gen = _random_epoch_gen(targets)
        if self._env_config.data_type == 'random-sort':
            self._target_gen = _sorted_data_gen(targets)

        self.target = None
        self.design = None
        self._folding = None
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
        # TODO: derivate state from dot-bracket target instead of local target for rna design with sequence constraints.
        # print('reset')
        self.target = next(self._target_gen)
        self.target.reset()
        if self._env_config.reward_function == 'structure_only':
            self.target.reset_counter()
        self.design = _Design(len(self.target))
        return self._get_state()

    def _apply_action(self, action):
        """
        Assign a nucleotide to a site.

        Args:
            action: The action chosen by the agent.
        """
        # print('apply action')
        current_site = self.design.first_unassigned_site if not self._env_config.reward_function == 'structure_only' else self.target.current_site
        paired_site = self.target.get_paired_site(current_site) if self._env_config.predict_pairs else None  # None for unpaired sites
        # print(current_site, paired_site)
        self.design.assign_sites(action, current_site, paired_site, self._env_config.predict_pairs)
        # print(self.design._primary_list)

    def _get_state(self):
        """
        Get a state dependend on the padded encoding of the target structure.

        Returns:
            The next state.
        """
        start = self.target.next_structure_site if self._env_config.reward_function == 'structure_only' else self.design.first_unassigned_site
        # print(f"get_state2: {self.target.structure_parts_encoding}")

        if not self._env_config.state_representation == 'n-gram':
            if self.design.last_assignment:
                index, value, paired_site = self.design.last_assignment
                self.target.assign_sites(index, value, paired_site)

        return self.target.padded_encoding[
            start : start + 2 * self._env_config.state_radius + 1
        ]

    def _get_local_design_loss(self, design):
        """
        Get the loss for the sequence parts and the structure part of the candidate solution.

        Args:
            design: Instance of the Design.

        Returns:
            distance: the total distance between the design and strutcure and sequence constraints.
            folding: the secondary structure to not compute this again.
        """
        distance = 0
        folding = fold(design.primary)[0] if self._env_config.reward_function == 'sequence_and_structure' else None
        sequence_parts, folding_parts = self.target.partition  # get_partition()  # return tuple of <sequence start, sequence end>

        if self._env_config.reward_function == 'sequence_and_structure':
            distance += hamming_with_n(design.primary, self.target.sequence_constraints)
        else:
            design = [c for c in design._primary_list]
            for index, site in enumerate(self.target.local_target):
                if site in ['A', 'C', 'G', 'U']:
                    design[index] = site
            self.design = _Design(primary=[c for c in ''.join(design).rstrip()])
            folding = fold(self.design.primary)[0]

        distance += hamming_with_n(folding, self.target.dot_bracket)

        return distance, folding


    def _local_improvement(self, folded_design):
        """
        Compute Hamming distance of locally improved candidate solutions.

        Returns:
            The minimum Hamming distance of all imporved candidate solutions.
        """
        def flatten_list(list_):
            return [item for sublist in list_ for item in sublist]

        if self._env_config.local_design:
            sequence_parts, structure_parts = self.target.partition
            differing_sites_per_part = []

            if self._env_config.reward_function == 'sequence_and_structure':
                for start, end in sequence_parts:
                    string_difference = _string_difference_indices(
                        self.target.sequence_constraints[start:end], self.design.primary[start:end]
                    )
                    string_difference = [index + start for index in string_difference]
                    differing_sites_per_part.append(string_difference)

            for start, end in structure_parts:
                string_difference = _string_difference_indices(
                    self.target.dot_bracket[start:end], self._folding[start:end]
                )
                string_difference = [index + start for index in string_difference]
                differing_sites_per_part.append(string_difference)

            differing_sites = flatten_list(differing_sites_per_part)
        else:

            differing_sites = _string_difference_indices(
                self.target.dot_bracket, folded_design
            )

        hamming_distances = []
        for mutation in product("AGCU", repeat=len(differing_sites)):
            mutated = self.design.get_mutated(mutation, differing_sites)
            folded_mutated, _ = fold(mutated.primary)
            if self._env_config.local_design:
                hamming_distance, _ = self._get_local_design_loss(mutated)
            else:
                hamming_distance = hamming_with_n(folded_mutated, self.target.dot_bracket)
            hamming_distances.append(hamming_distance)
            if hamming_distance == 0:  # For better timing results
                return 0
        return min(hamming_distances)

    def reward_local_design(self):
        """
        Reward implementation for partial RNA design.

        Returns:
            The reward.
        """
        distance, self._folding = self._get_local_design_loss(self.design)

        if distance == 0:
            return 1.0
        if 0 < distance < self._env_config.mutation_threshold:
            distance = self._local_improvement(self._folding)
        # distance might get larger than sites in target because of overlapping constraints when algorithm also places sequence
        if distance > len(self.target):
            return 0
        normalized_distance = (distance / len(self.target))

        episode_info = EpisodeInfo(
        target_id=self.target.id,
        time=time.time(),
        normalized_hamming_distance=normalized_distance,
        )
        self.episodes_info.append(episode_info)

        return (1 - normalized_distance) ** self._env_config.reward_exponent


    def _get_reward(self, terminal):
        """
        Compute the reward after assignment of all nucleotides.

        Args:
            terminal: Bool defining if final timestep is reached yet.

        Returns:
            The reward at the terminal timestep or 0 if not at the terminal timestep.
        """
        # print('reward')
        if not terminal:
            return 0

        if self._env_config.local_design:
            return self.reward_local_design()
        else:
            folded_design, _ = fold(self.design.primary)
            hamming_distance = hamming(folded_design, self.target.dot_bracket)
            if 0 < hamming_distance < self._env_config.mutation_threshold:
                hamming_distance = self._local_improvement(folded_design)

            normalized_hamming_distance = hamming_distance / len(self.target)

            # For hparam optimization
            episode_info = EpisodeInfo(
                target_id=self.target.id,
                time=time.time(),
                normalized_hamming_distance=normalized_hamming_distance,
            )
            self.episodes_info.append(episode_info)

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
        # print('execute')
        # print(actions)
        self._apply_action(actions)
        # print(f"execute: {self.target.structure_parts_encoding}")

        terminal = self.design.first_unassigned_site is None if not self._env_config.reward_function == 'structure_only' else all([x is None for x in self.target.structure_parts_encoding])
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
