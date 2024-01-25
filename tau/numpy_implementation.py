"""
Implementing abstract base classes on the basis of numpy arrays.
"""

from __future__ import annotations

from .base import Position, DialecticalStructure
from .numba_tau import direct_sub_arrays, gray, create_hash

from bitarray import bitarray
from itertools import product
from itertools import combinations
import numpy as np
from typing import List, Iterator, Set, Optional
from dd.autoref import BDD
from collections import deque


class NumpyPosition(Position):
    """Implements :py:class:`Position` on the basis of Numpy arrays."""

    def __init__(self, pos: np.ndarray):
        """Instantiates a :py:class:`NumpyPosition` from a numpy array.

        A sentence :math:`s_{i}` of a position :math:`\\mathcal{A}` (over a
        sentence pool with :math:`n` unnegated sentences) corresponds to an
        entry at index :math:`i-1` in a numpy array of length :math:`n`.
        :code:`1` represents acceptance (:math:`s_{i}\\in\\mathcal{A}`),
        :code:`2` stands for rejection (:math:`\\neg s_{i}\\in\\mathcal{A}`), and
        :code:`3` denotes acceptance and rejection (flat contradiction). Absent
        sentences correspond to :code:`0`. For example, for a sentence pool of 7
        unnegated sentences,
        :math:`\\mathcal{A} = \\lbrace s_{1}, \\neg s_{2}, s_{6} \\rbrace` correponds
        to :code:`numpy.array([1, 2, 0, 0, 0, 1, 0])`, which can be handed over
        to :py:class:`NumpyPosition`.

        .. note::
            It is more comfortable to instantiate :py:class:`NumpyPosition`
            with :py:class:`NumpyPosition.from_set`

        """
        self.__np_array = pos
        self.__hash = None
        self.__size = None
        self.n_unnegated_sentence_pool = len(pos)
        super(NumpyPosition, self).__init__()

    # representation as string when a Position is printed, for example
    def __repr__(self) -> str:
        return str(self.as_set())

    # hashing enables to form sets of Positions
    def __hash__(self):
        if not self.__hash:
            self.__hash = hash(create_hash(self.__np_array))
        return self.__hash

    def __eq__(self, other) -> bool:
        if isinstance(other, NumpyPosition):
            return np.array_equal(self.__np_array, other.__np_array)
        elif isinstance(other, Position):
            return super().__eq__(other)

    def __and__(self, other):
        return NumpyPosition(NumpyPosition.as_np_array(self) & NumpyPosition.as_np_array(other))

    def sentence_pool(self) -> Position:
        return NumpyPosition(np.ones(self.n_unnegated_sentence_pool))

    # number of non-supended sentences
    def size(self) -> int:
        if not self.__size:
            self.__size = np.count_nonzero(self.__np_array)
        return self.__size

    def domain(self) -> Position:
        arr = np.array([3 if self.__np_array[i] != 0 else 0 for i in range(len(self.__np_array))])

        return NumpyPosition(arr)

    # methods that return positions in different ways

    def as_bitarray(self) -> bitarray:
        pos = ''
        for i in range(self.n_unnegated_sentence_pool):
            if self.__np_array[i] == 0:
                pos += '00'
            elif self.__np_array[i] == 1:
                pos += '10'
            elif self.__np_array[i] == 2:
                pos += '01'
            else:
                pos += '11'
        return bitarray(pos)

    @staticmethod
    def to_numpy_position(position: Position) -> NumpyPosition:
        """Convert an implemented instance of :py:class:`Position` to an \
        instance of :py:class:`NumpyPosition`.
        """

        if isinstance(position, NumpyPosition):
            return position
        return NumpyPosition.from_set(position.as_set(), position.sentence_pool().size())

    @staticmethod
    def as_np_array(position: Position) -> np.ndarray:
        """Position as numpy array.

        :param position: An instance of :py:class:`Position`

        :returns: A numpy array of length :math:`n` (size of the sentence pool \
        associated with the position). An entry at index :math:`i` corresponds \
        to the standing of :math:`s_{i+1}` according to the following \
        conventions: :code:`1`: acceptance, :code:`2`: rejection, \
        :code:`3`: acceptance and rejection, :code:`0`: abstention
        """
        if isinstance(position, NumpyPosition):
            return position.__np_array
        else:
            arr = np.zeros(position.sentence_pool().size())
            for s in position.as_set():
                if s < 0:
                    arr[abs(s) - 1] += 2
                elif s > 0:
                    arr[s - 1] += 1
            return arr

    def as_ternary(self) -> int:
        if self.is_minimally_consistent():
            return sum(self.__np_array[i] * 10 ** i for i in range(len(self.__np_array)))
        else:
            return None

    def as_set(self) -> Set[int]:

        arr = self.__np_array.astype(np.int32)

        pos = set()
        for i in range(len(arr)):
            if arr[i] in {1, 2}:
                pos.add((i + 1) * ((-1) ** (arr[i] - 1)))
            elif arr[i] == 3:
                pos.add(i + 1)
                pos.add(-(i + 1))

        # convert elements from np.int32 to int for JSON serializability
        pos = set(int(s) for s in pos)

        return pos

    @staticmethod
    def from_set(position: Set[int], n):
        arr = np.zeros(n)
        for s in position:
            if s < 0:
                arr[abs(s) - 1] += 2
            elif s > 0:
                arr[s - 1] += 1
        return NumpyPosition(arr)

    def as_list(self) -> List[int]:
        return list(self.as_set())

    def is_minimally_consistent(self) -> bool:
        return 3 not in self.__np_array

    def are_minimally_compatible(self, position: Position) -> bool:
        return self.union([self, position]).is_minimally_consistent()

    def is_subposition(self: Position, pos2: Position) -> bool:
        return self.as_set().issubset(pos2.as_set())

    def subpositions(self, n: int = -1, only_consistent_subpositions: bool = True) -> Iterator[Position]:
        if n == -1:
            # by default, create set of all subpositions:
            res = set()
            for i in range(len(self.__np_array) + 1):
                res.update(self.subpositions(i, only_consistent_subpositions))

            return iter(res)

        else:
            subpositions = [NumpyPosition.from_set(pos, len(self.__np_array))
                            for pos in combinations(self.as_set(), n)]

            if only_consistent_subpositions and not self.is_minimally_consistent():
                subpositions = [position for position in subpositions if position.is_minimally_consistent()]

            return iter(subpositions)

    def direct_subpositions(self) -> Iterator[Position]:
        """Iterate over subpositions that have exactly one element less than the
        position itself.
        """

        return iter([NumpyPosition(arr) for arr in direct_sub_arrays(self.__np_array, len(self.__np_array))])

    def is_accepting(self, sentence: int) -> bool:

        if not self.size():  # self is the empty position
            return False

        if sentence < 0:  # rejected sentence
            return self.__np_array[abs(sentence) - 1] == 2

        else:  # accepted sentence
            return self.__np_array[abs(sentence) - 1] == 1

    def is_in_domain(self, sentence: int) -> bool:
        try:
            return self.__np_array[abs(sentence) - 1] != 0
        except IndexError:
            return False

    @staticmethod
    def union(positions: Set[Position]) -> Position:
        if not positions:
            return NumpyPosition.from_set(set(), 0)
        # assumption: positions have the same length
        if len(positions) == 1:
            return next(iter(positions))
        else:
            position_list = list(NumpyPosition.as_np_array(pos) for pos in positions)
            n = len(position_list[0])
            union = np.zeros(n)

            for i in range(n):
                if (any(1.0 == pos[i] for pos in position_list)
                        and any(2.0 == pos[i] for pos in position_list)):
                    union[i] = 3
                else:
                    union[i] = max(pos[i] for pos in position_list)

            return NumpyPosition(union)

    @staticmethod
    def intersection(positions: Set[Position]) -> Position:

        if not positions:
            return NumpyPosition(np.array([]))
        elif len(positions) == 1:
            return next(iter(positions))
        else:
            position_list = list(NumpyPosition.as_np_array(pos) for pos in positions)
            n = len(position_list[0])
            intersection = np.zeros(n)

            for i in range(n):
                if all(position_list[0][i] == pos[i] for pos in position_list[1:]):
                    intersection[i] = position_list[0][i]
            return NumpyPosition(intersection)

    # ToDo: Perhaps only temporarily a public static method (at the moment used by other Position classes to
    # quick and dirty implement `neighbours`).
    @staticmethod
    def np_neighbours(position: Position, depth: int) -> Iterator[np.ndarray]:
        # generate variations of a position's Numpy array representation by changing at most depth elements
        np_position = NumpyPosition.as_np_array(position)

        queue = deque()
        queue.append((np_position, 0, depth))

        while queue:

            vertex, level, changes_left = queue.popleft()

            if not changes_left or level == len(vertex):
                yield vertex

            if changes_left and level < len(vertex):

                for v in [0, 1, 2]:
                    neighbour = vertex.copy()
                    neighbour[level] = v
                    if v == vertex[level]:  # nothing changed
                        queue.append((neighbour, level + 1, changes_left))
                    else:
                        queue.append((neighbour, level + 1, changes_left - 1))

    def neighbours(self, depth: int) -> Iterator[Position]:
        """Iterate over the neighbours of the position.

        :param depth: Depth of the neighbourhood, i.e. positions that can be \
        reached by at most :code:`depth` many adjustments of individual sentences.\
        The number of neighbours is :math:`\\sum_{k=0}^{d} {n \\choose k} \\cdot 2^k`,\
        where :math:`n` is the number of unnegated sentences in the sentence pool,\
        and :math:`d` is the depth of the neighbourhood, including the position itself.

        :returns: An iterator over positions in the neighbourhood of the position,\
        including the position itself.
        """

        for neighbour in NumpyPosition.np_neighbours(self, depth):
            yield NumpyPosition(neighbour)


class DAGNumpyDialecticalStructure(DialecticalStructure):
    """Implementing :py:class:`DialecticalStructure` on the basis of
    :py:class:`NumpyPosition` and directed acyclic graphs (DAG).

    .. note::
        This class calculates and stores information about *all* positions in a \
        dialectical structure. The number of positions grows exponentially in \
        the number of sentences in the sentence pool (:math:`3^{n}`). \
        Consequently, it is not suitable for larger sentence pool sizes, for \
        which :py:class:`BDDNumpyDialecticalStructure` is recommended.

    """

    def __init__(self, n: int, initial_arguments: List[List[int]] = None, name: str = None):
        self.arguments_cnf = set()
        self.arguments = []
        self.name = name
        self.n = n  # number of unnegated sentences in sentence pool used to iterate through positions
        self.__sp = NumpyPosition(np.ones(n))  # full sentence pool

        # prepare storage for results of costly functions
        self.__cons_comp_pos = set()
        self.__consistent_parents = {}
        self.__complete_consistent_extensions = {}
        self._closures = {}
        self.__min_cons_pos = set()

        ###
        self.__consistent_extensions = {}
        self.__dict_n_complete_extensions = {}
        self.__n_extensions = {}

        # update status
        self.__n_updates = 0
        self.__updated = False
        if initial_arguments:
            for arg in initial_arguments:
                self.add_argument(arg)

            # update the dialectical structure
            self._update()

    @staticmethod
    def from_arguments(arguments: List[List[int]], n_unnegated_sentence_pool: int,
                       name: str = None) -> DialecticalStructure:
        return DAGNumpyDialecticalStructure(n_unnegated_sentence_pool, arguments, name)

    def get_name(self) -> str:
        return self.name

    def set_name(self, name: str):
        self.name = name

    def add_argument(self, argument: List[int]) -> DialecticalStructure:
        # arguments are represented as a single position (of its negated premises + its conclusion)
        # as it stands, the original arguments are not retrievable from this representation

        pos = np.zeros(self.n, dtype=int)
        for s in argument[:-1]:
            pos[abs(s) - 1] = 1 if s < 0 else 2

        pos[abs(argument[-1]) - 1] = 1 if argument[-1] > 0 else 2

        self.arguments_cnf.add(NumpyPosition(pos))
        self.arguments.append(argument)
        # If new arguments are added to a dialectical structure, already existing
        # objects such as the DAG or the closures may need to be recreated. The need
        # for an update is indicated by setting its update status to False.
        self.__updated = False

        return self

    def add_arguments(self, arguments: List[List[int]]) -> DialecticalStructure:
        for argument in arguments:
            self.add_argument(argument)
        self._update()
        return self

    def get_arguments(self) -> List[List[int]]:
        # remove tautological arguments (which are an artifact of this implementation)
        arguments = self.arguments.copy()
        for s in range(1, self.sentence_pool().size() + 1):
            if [s, s] in arguments:
                arguments.remove([s, s])
        return arguments

    # Sentence-pool, domain and completeness

    def is_complete(self, position: Position) -> bool:
        return self.__sp.domain() == position.domain()

    def sentence_pool(self) -> Position:
        return self.__sp

    # Dialectic consistency & compatibility

    def _satisfies(self, argument: Position, position: Position) -> bool:

        """Check whether a complete `position` satisfies `argument`, which is also
        given as a position (negated premises + conclusion). Satisfiability amounts
        to checking whether bitwise AND operation on the numpy arrays associated with
        the positions is non-zero at least once.
        """

        return bool(np.count_nonzero(NumpyPosition.as_np_array(argument)
                                     & NumpyPosition.as_np_array(position)))

    def is_consistent(self, position: Position) -> bool:
        # check update status of dialectical structure
        self._update()

        return NumpyPosition.to_numpy_position(position) in self.__complete_consistent_extensions

    def are_compatible(self, position1: Position, position2: Position, ) -> bool:
        # check update status of dialectical structure
        self._update()

        # at least one of the positions is not minimally consistent
        if not (position1.is_minimally_consistent() and position2.is_minimally_consistent()):
            return False

        # at least one of the positions is not dialectically consistent
        elif not (self.is_consistent(position1) and self.is_consistent(position2)):
            return False

        else:
            # do they have a complete and consistent extension in common?
            return any(self.__complete_consistent_extensions[NumpyPosition.to_numpy_position(position1)].intersection(
                self.__complete_consistent_extensions[NumpyPosition.to_numpy_position(position2)]))

    def minimally_consistent_positions(self) -> Iterator[Position]:

        if self.__min_cons_pos:
            return iter(self.__min_cons_pos)
        else:
            # create the set of all positions
            min_consistent_positions = set()

            for pos in gray(3, self.n):
                min_consistent_positions.add(NumpyPosition(pos.copy()))

            self.__min_cons_pos = min_consistent_positions
            return iter(min_consistent_positions)

    def consistent_positions(self) -> Iterator[Position]:
        # check update status of dialectical structure
        self._update()

        # note that the complete_parent_graph and direct_parent_graph have the
        # same keys
        return iter(self.__complete_consistent_extensions)

    def consistent_complete_positions(self) -> Iterator[Position]:
        # self.__update()
        if not self.__cons_comp_pos:
            all_complete_positions = [NumpyPosition(np.array(e))
                                      for e in product([1, 2], repeat=self.n)]

            # filter positions that do not satisfy all arguments:
            self.__cons_comp_pos = set(
                [pos for pos in all_complete_positions
                 if all(self._satisfies(arg, pos) for arg in self.arguments_cnf)])

            return iter(self.__cons_comp_pos)

        else:
            # check update status of dialectical structure
            # self.check_update()

            return iter(self.__cons_comp_pos)

    # Dialectic entailment and dialectic closure of consistent positions

    def entails(self, position1: Position, position2: Position) -> bool:
        # check update status of dialectical structure
        self._update()

        if not self.is_consistent(position1):  # ex falso quodlibet
            return True
        elif not self.is_consistent(position2):
            return False

        # catch complete positions (else-statements)
        if self.__complete_consistent_extensions[NumpyPosition.to_numpy_position(position1)]:
            pos1_extensions = self._complete_extensions(position1)
        else:
            pos1_extensions = {NumpyPosition.to_numpy_position(position1)}

        if self.__complete_consistent_extensions[NumpyPosition.to_numpy_position(position2)]:
            pos2_extensions = self._complete_extensions(position2)
        else:
            pos2_extensions = {NumpyPosition.to_numpy_position(position2)}

        return pos1_extensions.issubset(pos2_extensions)

    def closure(self, position: Position) -> Position:
        if self.is_consistent(position):
            return self._closures[NumpyPosition.to_numpy_position(position)]
        # ex falso quodlibet
        else:
            return self.__sp.domain()

    def is_closed(self, position: Position) -> bool:
        # check update status of dialectical structure
        self._update()

        return position == self.closure(position)

    def closed_positions(self) -> Iterator[Position]:
        self._update()
        return iter([pos for pos in self.consistent_positions() if self.is_closed(pos)])

    def axioms(self, position: Position,
               source: Iterator[Position] = None) -> Iterator[Position]:

        self._update()

        if not self.is_consistent(position):
            raise ValueError("An inconsistent Position cannot be axiomatized!")

        res = set()

        # if no source is provided, default to all consistent positions
        if not source:
            source = self.consistent_positions()

        #  collect minimal positions from *source*, which entail *position*
        for pos in source:
            if (self.entails(pos, position)
                    and not any(self.entails(subpos, position) for subpos in pos.subpositions() if subpos != pos)):
                res.add(pos)

        if not res:
            return []

        return iter(res)

    def is_minimal(self, position: Position) -> bool:
        for pos in position.subpositions():
            if pos != position and self.entails(pos, position):
                return False
        return True

    def minimal_positions(self) -> Iterator[Position]:

        res = set()
        for pos in self.consistent_positions():
            if self.is_minimal(pos):
                res.add(pos)
        return iter(res)

    # Auxiliary method to retrieve complete extension from the graph
    def _complete_extensions(self, position: Position) -> Set[Position]:
        """Returns complete extensions of `position` by retrieving corresponding
         node in the graph that stores complete extensions."""

        # check update status of dialectical structure
        self._update()

        if not position.is_minimally_consistent():
            return set()
        # complete positions have no parents but extend themselves
        elif len(self.__complete_consistent_extensions[NumpyPosition.to_numpy_position(position)]) == 0:
            return {position}
        else:
            return self.__complete_consistent_extensions[NumpyPosition.to_numpy_position(position)]

    def n_complete_extensions(self, position: Position = None) -> int:
        self._update()
        if not position or position.size() == 0:
            return len(self.__cons_comp_pos)
        else:
            return len(self._complete_extensions(position))

    # conditional degree of justification
    def degree_of_justification(self, position1: Position, position2: Position) -> float:

        return len(self._complete_extensions(position1).intersection(
            self._complete_extensions(position2))) / self.n_complete_extensions(position2)

    def _update(self) -> None:
        """Checks whether the dialectical structure is up to date,
        which may not be the case after new arguments have been added after its
        initialisation. If the structure is outdated, this method resets and
        recreates stored objects (e.g. DAG, closure). All methods that require
        an updated structure to work properly, call this method.
        """

        if not self.__updated:
            self.__complete_consistent_extensions = {}
            self.__consistent_extensions = {}
            self.__consistent_parents = {}

            self.__dict_n_complete_extensions = {}
            self.__n_extensions = {}
            self._closures = {}

            # Minimally consistency is not affected by structural updates,
            # but it is created if it does not already exist.
            if not self.__min_cons_pos:
                self.minimally_consistent_positions()

            # starting point (gamma_n): consistent complete positions
            current_gamma = set(self.consistent_complete_positions())

            for pos in current_gamma:
                # complete positions have no parents
                self.__complete_consistent_extensions[pos] = {pos}
                self.__consistent_parents[pos] = set()
                self.__consistent_extensions[pos] = {pos}
                self.__dict_n_complete_extensions[pos] = 1
                self._closures[pos] = pos
                self.__n_extensions[pos] = 1

            new_gamma = set()
            # iterate backwards over length of subpositions
            for _ in reversed(range(0, self.n)):
                for position in current_gamma:

                    for sub_pos in position.direct_subpositions():

                        # add parent to dictionary
                        if sub_pos not in self.__complete_consistent_extensions.keys():
                            self.__complete_consistent_extensions[sub_pos] = \
                                set(self.__complete_consistent_extensions[position])
                            self.__consistent_parents[sub_pos] = {position}
                            self.__consistent_extensions[sub_pos] = {sub_pos}
                            self.__consistent_extensions[sub_pos].update(self.__consistent_extensions[position])

                            # self.closures[sub_pos] = self.closures[position]

                        else:
                            self.__consistent_parents[sub_pos].add(position)
                            self.__consistent_extensions[sub_pos].update(self.__consistent_extensions[position])
                            self.__complete_consistent_extensions[sub_pos].update(
                                set(self.__complete_consistent_extensions[position]))

                            # self.closures[sub_pos] &= self.closures[position]

                        new_gamma.add(sub_pos)

                for pos in new_gamma:

                    # Gregor's Condition

                    # Gregor's Condition
                    for par in self.__consistent_parents[pos]:
                        if len(self.__complete_consistent_extensions[pos]) == \
                                len(self.__complete_consistent_extensions[par]):
                            self._closures[pos] = self._closures[par]
                            break
                    if pos not in self._closures:
                        self._closures[pos] = pos

                current_gamma = new_gamma
                new_gamma = set()

        # update status of dialectical structure
        self.__updated = True

        return None


class BDDNumpyDialecticalStructure(DAGNumpyDialecticalStructure):
    """Implementing :py:class:`DialecticalStructure` on the basis of
    :py:class:`NumpyPosition` and binary decision diagrams (BDD).
    """

    name: Optional[str]

    def __init__(self, n: int, initial_arguments: List[List[int]] = None, name: str = None):
        """Initialize an instance of :py:class:`BDDNumpyDialecticalStructure`."""

        self.__updated = False
        self.__full_sentence_pool = NumpyPosition(np.array([3 for i in range(n)]))
        super().__init__(n, initial_arguments, name)

        # add trivial arguments to catch sentences that are not used in any argument
        for s in range(1, n + 1):
            self.arguments.append([s, s])

    @staticmethod
    def from_arguments(arguments: List[List[int]], n_unnegated_sentence_pool: int,
                       name: str = None) -> DialecticalStructure:
        return BDDNumpyDialecticalStructure(n_unnegated_sentence_pool, arguments, name)

    def get_name(self) -> str:
        return self.name

    def set_name(self, name: str):
        self.name = name

    # auxiliary methods
    def _args_to_expr(self):
        u = ''
        # argument: disjunction of conclusion and negated premises
        for arg in self.arguments:
            conclusion = arg[-1]
            if conclusion < 0:
                expr = '~s{}'.format(abs(conclusion))
            else:
                expr = 's{}'.format(conclusion)
            for prem in arg[:-1]:
                if prem < 0:
                    expr += ' | s{}'.format(abs(prem))
                else:
                    expr += ' | ~s{}'.format(prem)
            # conjoin arguments
            expr = '( {} ) & '.format(expr)
            u += expr
        # remove trailing characters
        u = u[:-3]
        return u

    def _pos_to_expr(self, position: NumpyPosition):
        """Convert a position to an expression suitable for bdd."""

        expr = ''
        for s in position.as_set():
            if s < 0:
                expr += '~s{} & '.format(abs(s))
            else:
                expr += 's{} & '.format(s)
        expr = expr[:-3]
        return expr

    def _dict_to_set(self, d: dict):
        s = set()
        for key, value in d.items():
            key = int(key[1:])
            if value:  # value is True
                s.add(key)
            else:
                s.add(-key)
        return s

    def _update(self) -> None:
        if self.__updated:
            return

        # represent dialectical structure as bdd
        self.bdd = BDD()

        # declare variables for every sentence from sentence pool
        self.bdd.declare(*['s{}'.format(j) for j in range(1, self.n + 1)])

        # add expression representing dialectical structure
        self.dia_expr = self.bdd.add_expr(self._args_to_expr())

        self.__updated = True

    # ToDo: This method logs a lot of WARNINGs from DD. Perhaps, we should implement it differently.
    # So far, the logs have to be supressed by configuring the DD-logger via
    # logging.getLogger('dd').setLevel(logging.ERROR)
    # Andreas' input:
    # "Die WARNINGs von dd.BDD (der Art "WARNING:dd.bdd:Missing bits:  support - care_vars =
    # {'s5', 's2', 's6', 's1', 's4', 's3'}") entstehen in der BDDNumpyDialecticalStructure-Methode
    # "closure" (aus rethon/numpy_implementation.py). Dort erhält ".bdd.pick.iter" eine leere Liste
    # als "care_vars". In meiner Erinnerung hat sich das als sinnvolle Optimierung herausgestellt:
    # Wenn wir als care_vars zum Beispiel alle Sätze des Satzpools angeben, verschwindet die Warnung,
    # aber es werden viel mehr Modelle (für uns: vollständig und konsistente Positionen) zurückgegeben,
    # die anschliessend miteinander geschnitten werden müssen.
    # Ich sehe im Moment keine schnelle Lösung ausser das log-level von INFO auf ERROR zu setzen,
    # aber vielleicht gibt es in der Methode "closure" ungeahntes Verbesserungspotential.
    def closure(self, position: Position) -> Position:

        # the position's closure has been calculated before
        if position in self._closures:
            return self._closures[position]

        if position.size() == 0:  # empty position
            models = list(self.bdd.pick_iter(self.dia_expr, care_vars=[]))
        else:
            # convert position to bdd expression
            v = self.bdd.add_expr(self._pos_to_expr(position))

            # models: extensions of position with variblses occuring along the recursive traversal of the BDD
            models = list(self.bdd.pick_iter(self.dia_expr & v, care_vars=[]))

        if not models:
            return self.__full_sentence_pool

        # intersect models
        closure = dict(set.intersection(*(set(d.items()) for d in models)))

        closure = self._dict_to_set(closure)
        closure = NumpyPosition.from_set(closure, self.n)

        # store closure for later reuse
        self._closures[position] = closure

        return closure

    def is_consistent(self, position: NumpyPosition) -> bool:
        # check update status of dialectical structure
        self._update()

        return self.closure(position) != self.__full_sentence_pool

    def are_compatible(self, position1: Position, position2: Position, ) -> bool:
        # check update status of dialectical structure
        self._update()

        # case: at least one of the positions is not minimally consistent
        if not (position1.is_minimally_consistent() and position2.is_minimally_consistent()):
            return False
        # case: both positions are minimally consistent
        else:
            pos_union = NumpyPosition.union({position1, position2})
            return self.closure(pos_union) != self.__full_sentence_pool
            # if not self.closure(pos_union):
            #    return False
            # return True

    def minimally_consistent_positions(self) -> Iterator[Position]:
        for position in gray(3, self.n):
            yield NumpyPosition(position.copy())

    def consistent_positions(self) -> Iterator[Position]:
        # check update status of dialectical structure
        self._update()
        empty_pos = NumpyPosition.from_set(set(), self.n)
        for neighbour in empty_pos.neighbours(self.n):
            if self.is_consistent(neighbour):
                yield neighbour
        # raise NotImplementedError("Cannot iterate over all consistent positions.")

    def consistent_complete_positions(self) -> Iterator[Position]:
        self._update()

        models = list(self.bdd.pick_iter(self.dia_expr, care_vars=self.bdd.vars))

        for model in models:
            position = self._dict_to_set(model)
            position = NumpyPosition.from_set(position, self.n)
            yield position

    def entails(self, position1: Position, position2: Position) -> bool:
        # check update status of dialectical structure
        self._update()

        if not self.is_consistent(position1):  # ex falso quodlibet
            return True
        elif not self.is_consistent(position2):
            return False

        closure_pos1 = self.closure(position1)

        # position1 is dialectically inconsistent:
        if not closure_pos1:
            return False

        return position2.as_set().issubset(closure_pos1.as_set())

    def axioms(self, position: Position,
               source: Iterator[Position] = None) -> Iterator[Position]:
        position = NumpyPosition.to_numpy_position(position)
        self._update()

        if not self.is_consistent(position):
            raise ValueError("An inconsistent Position cannot be axiomatized!")

        res = set()

        # old: if no source is provided, default to all consistent positions of dialectical structure
        # here: default to all dialectically consistetn neighbours (full range). That is risky for large sentence pools.
        # ToDo: Discuss.
        if not source:
            # raise NotImplementedError("Cannot iterate over all consistent positions.")
            # source = position.neighbours(self.n) (we should exclude dial. inconsistent neighbours)
            source = self.consistent_positions()
        #  collect inclusion minimal positions from *source*, which entail *position*
        for pos in source:
            if (self.entails(pos, position)
                    and not any(self.entails(subpos, position) for subpos in pos.subpositions() if subpos != pos)):
                res.add(pos)
        # if nothing from the source entails position, return None
        if not res:
            return []

        return iter(res)

    # def minimal_positions(self) -> Iterator[Position]:
    #    raise NotImplementedError("Cannot iterate over all consistent positions.")

    def n_complete_extensions(self, position: Position = None) -> int:
        self._update()
        if not position or position.size() == 0:
            return self.bdd.count(self.dia_expr, nvars=self.n)
        else:
            v = self.bdd.add_expr(self._pos_to_expr(position))
            return self.bdd.count(self.dia_expr & v, nvars=self.n)

    def degree_of_justification(self, position1: Position, position2: Position) -> float:

        v1 = self.bdd.add_expr(self._pos_to_expr(position1))
        v2 = self.bdd.add_expr(self._pos_to_expr(position2))

        a = self.bdd.count(self.dia_expr & v1 & v2, nvars=self.n)
        b = self.bdd.count(self.dia_expr & v2, nvars=self.n)

        return a / b
