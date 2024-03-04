"""
Implementing abstract base classes on the basis of Python sets.
"""

# see: https://stackoverflow.com/questions/33533148
from __future__ import annotations

from .base import Position, DialecticalStructure

from bitarray import bitarray
from typing import Set, Iterator, List
from itertools import chain, combinations
from pysat.formula import CNF
from pysat.solvers import Minisat22
import numpy as np
from collections import deque

class SetBasedPosition(Position):
    """An implementation of :py:class:`Position` on the basis of Python sets.
    """

    # comment: `def __init__(self, position1: Set[int] = set()):` should do as well, but it doesn't... beats me
    def __init__(self, position: Set[int], n_unnegated_sentence_pool: int):
        self.__position = frozenset(position)
        self.n_unnegated_sentence_pool = n_unnegated_sentence_pool
        super().__init__()


    # representation as string when a Position is printed, for example
    def __repr__(self) -> str:
        return set(self.__position).__repr__()

    def __iter__(self):
        return self.__position.__iter__()

    def __next__(self):
        return self.__position.__next__()

    def __raise_value_error_if_sentence_pool_mismatch(self, position: Position):
        if self.sentence_pool() != position.sentence_pool():
            raise ValueError("The function you called expects positions to be based on the same sentence pool as",
                             " the dialectical structure.")

    @staticmethod
    def from_set(position: Set[int], n_unnegated_sentence_pool: int) -> Position:
        return SetBasedPosition(position, n_unnegated_sentence_pool)

    @staticmethod
    def as_setbased_position(position: Position) -> SetBasedPosition:
        if isinstance(position, SetBasedPosition):
            return position
        else:
            return SetBasedPosition(position.as_set(),
                                    position.sentence_pool().size())

    def sentence_pool(self) -> Position:
        return SetBasedPosition.from_set(set(range(1, self.n_unnegated_sentence_pool + 1)),
                                         self.n_unnegated_sentence_pool)

    def domain(self) -> Position:
        return SetBasedPosition(self.__position | {-1 * sentence for sentence in self.__position},
                                self.sentence_pool().size())

    def as_bitarray(self) -> bitarray:
        #if len(self.__position) == 0:
        #    return bitarray()

        position_ba = 2 * self.n_unnegated_sentence_pool * bitarray('0')
        #if n_unnegated_sentence_pool:
        #    position_ba = 2 * n_unnegated_sentence_pool * bitarray('0')
        #elif len(self.__position) == 0:
        #    return bitarray()
        #else:
        #    position_ba = 2 * max([abs(sentence) for sentence in self.__position]) * bitarray('0')
        for sentence in self.__position:
            if sentence < 0:
                position_ba[2 * (abs(sentence) - 1) + 1] = True
            else:
                position_ba[2 * (abs(sentence) - 1)] = True
        return position_ba

    def as_ternary(self) -> int:
        #if self.is_minimally_consistent():
        res = 0
        for sentence in self.sentence_pool():
            if sentence in self.__position:
                if -sentence in self.__position:
                    res += 3 * 10 ** (sentence - 1)
                else:
                    res += 1 * 10 ** (sentence - 1)
            elif -sentence in self.__position:
                res += 2 * 10 ** (sentence - 1)
            # for sentence in self.__position:
            #     if sentence < 0:
            #         res += 2 * 10 ** (abs(sentence) - 1)
            #     else:
            #         res += 1 * 10 ** (sentence - 1)
        return res
        #else:
        #    return None

    def as_set(self) -> Set[int]:
        return set(self.__position)

    def as_list(self) -> List[int]:
        return list(self.__position)

    def is_minimally_consistent(self) -> bool:
        return not any([-1 * element in self.__position for element in self.__position])

    def is_minimally_compatible(self, position: Position) -> bool:
        self.__raise_value_error_if_sentence_pool_mismatch(position)
        return SetBasedPosition(self.as_set() | position.as_set(),
                                max(self.sentence_pool().size(),
                                    position.sentence_pool().size())).is_minimally_consistent()

    def is_subposition(self, position: Position) -> bool:
        self.__raise_value_error_if_sentence_pool_mismatch(position)
        return self.__position.issubset(position.as_set())

    def subpositions(self, n: int = -1, only_consistent_subpositions: bool = True) -> Iterator[Position]:
        if self.is_minimally_consistent() or only_consistent_subpositions == False:
            # for n = -1 or invalid n-values just return the powerset of position
            if n == -1:
                return chain.from_iterable(
                    map(lambda x: SetBasedPosition(set(x), self.sentence_pool().size()),
                        combinations(list(self.__position), r)) for r in
                    range(len(self.__position) + 1))
            else:
                return map(lambda x: SetBasedPosition(set(x), self.sentence_pool().size()),
                           combinations(list(self.__position), n))
        else:
            # for n = -1 just return the powerset of position
            if n == -1:
                return filter(lambda x: x.is_minimally_consistent(), chain.from_iterable(
                    map(lambda x: SetBasedPosition(set(x), self.sentence_pool().size()),
                        combinations(list(self.__position), r)) for r in
                    range(len(self.__position) + 1)))
            else:
                return filter(lambda x: x.is_minimally_consistent(),
                                   map(lambda x: SetBasedPosition(set(x), self.sentence_pool().size()),
                                       combinations(list(self.__position), n)))

    def is_accepting(self, sentence: int) -> bool:
        return sentence in self.__position

    def is_in_domain(self, sentence: int) -> bool:
        return sentence in self.domain().as_set()

    def size(self) -> int:
        return len(self.__position)

    def union(self, *positions: Position) -> Position:
        if not positions:
            return self
        if len({self.sentence_pool().size()} | {pos.sentence_pool().size() for pos in positions}) != 1:
            raise ValueError("Union of positions is restricted to positions with matching sentence pools.")
        ret = self.as_set()
        for position in positions:
            ret = ret | position.as_set()
        return SetBasedPosition(list(ret), self.sentence_pool().size())

    # operator version of union
    def __or__(self, other):
        if isinstance(other, Position):
            return self.union(other)
        else:
            raise TypeError(f"{other} must be a theodias.Position.")

    def intersection(self, *positions: Position) -> Position:
        if not positions:
            return SetBasedPosition(set(), self.sentence_pool().size())
        if len({self.sentence_pool().size()} | {pos.sentence_pool().size() for pos in positions}) != 1:
            raise ValueError("Intersection of positions is restricted to positions with matching sentence pools.")
        ret = self.as_set()
        for position in positions:
            ret = ret & position.as_set()
        return SetBasedPosition(list(ret), self.sentence_pool().size())

    # operator version of intersection
    def __and__(self, other):
        if isinstance(other, Position):
            return self.intersection(other)
        else:
            raise TypeError(f"{other} must be a theodias.Position.")

    def difference(self, other: Position ) -> Position:
        if self.sentence_pool().size() != other.sentence_pool().size():
            raise ValueError("Difference of positions is restricted to positions with matching sentence pools.")
        return SetBasedPosition(self.as_set().difference(other.as_set()), self.sentence_pool().size())

    # operator version of difference
    def __sub__(self, other):
        if isinstance(other, Position):
            return self.difference(other)
        else:
            raise TypeError(f"{other} must be a theodias.Position.")

    def neighbours(self, depth: int) -> Iterator[Position]:

        position_array = [0] * self.sentence_pool().size()

        for s in self.as_set():
            if s > 0:
                position_array[abs(s) - 1] = 1
            elif s < 0:
                position_array[abs(s) - 1] = 2

        queue = deque()
        queue.append((position_array, 0, depth))

        while queue:
            vertex, level, changes_left = queue.popleft()

            if not changes_left or level == len(vertex):

                neighbour_set = set((i + 1) * (-1) ** (vertex[i] - 1)
                                    for i in range(len(vertex)) if vertex[i] != 0)

                yield SetBasedPosition(neighbour_set, self.sentence_pool().size())

            if changes_left and level < len(vertex):

                for v in [0, 1, 2]:
                    neighbour = vertex.copy()
                    neighbour[level] = v
                    if v == vertex[level]:  # nothing changed
                        queue.append((neighbour, level + 1, changes_left))
                    else:
                        queue.append((neighbour, level + 1, changes_left - 1))

class DAGSetBasedDialecticalStructure(DialecticalStructure):
    """An implementation of :py:class:`DialecticalStructure` on the basis of :py:class:`SetBasedPosition`.

        .. note::

            This implementations is a reference implementation, which is not optimized for performance. We advice to
            use :py:class:`DAGNumpyDialecticalStructure` or :py:class:`BDDNumpyDialecticalStructure`
            in non-illustrative contexts.
    """

    def __init__(self, n: int, initial_arguments: List[List[int]] = None, name: str = None):
        self.arguments = []
        self.n = n  # number of unnegated sentences in sentence pool used to iterate through positions
        self.__sentence_pool = [i for i in range(-n, n + 1) if not i == 0]
        self.name = name

        # initialise here in the case the ds is empty
        self.cnf = CNF()

        if initial_arguments:
            for arg in initial_arguments:
                self.add_argument(arg)
        self.__dirty = True
        #self.__update()

    @staticmethod
    def from_arguments(arguments: List[List[int]], n_unnegated_sentence_pool: int,
                       name: str = None ) -> DialecticalStructure:
        return DAGSetBasedDialecticalStructure(n_unnegated_sentence_pool, arguments, name)

    def get_name(self) -> str:
        return self.name

    def set_name(self, name: str):
        self.name = name

    @staticmethod
    def __argument_to_disjunction(argument: List[int]):
        return [-1 * argument[i] if i < len(argument) - 1 else argument[i] for i in range(len(argument))]

    def __cnf(self, arguments) -> CNF:
        cnf = CNF()
        for clause in [DAGSetBasedDialecticalStructure.__argument_to_disjunction(argument) for argument in arguments]:
            cnf.append(clause)
        # flat list of all sentences appearing in arguments
        cnf_sentences = {item for sublist in cnf.clauses for item in sublist}
        # sentences from the sentence pool if neither they nor their negation appear in arguments
        missing_sentences = [sentence for sentence in self.__sentence_pool if
                             sentence not in cnf_sentences and -sentence not in cnf_sentences]
        # adding those as disjunction to the cnf
        if missing_sentences:
            cnf.append(missing_sentences)
        return cnf

    def compute_complete_consistent_positions(self):
        m = Minisat22(self.cnf.clauses)
        gamma = m.enum_models()
        return [SetBasedPosition(pos, self.n) for pos in gamma]

    def __update(self):
        if self.__dirty:
            self.__complete_consistent_extensions = {}
            # These two I call only in the constructor
            self.__consistent_extensions = {}
            self.__consistent_parents = {}

            self.__dict_n_complete_extensions = {}
            self.__n_extensions = {}
            self.__closures = {}

            if self.arguments:

                self.complete_consistent_positions = self.compute_complete_consistent_positions()
                current_gamma = self.complete_consistent_positions
                for pos in current_gamma:
                    # complete positions have no parents
                    self.__complete_consistent_extensions[pos] = {pos}
                    self.__consistent_extensions[pos] = {pos}
                    self.__dict_n_complete_extensions[pos] = 1
                    self.__closures[pos] = pos
                    self.__n_extensions[pos] = 1

                new_gamma = set()
                # iterate backwards over length of subpositions
                for i in reversed(range(0, self.n)):
                    for position in current_gamma:
                        for sub_pos in combinations(position, i):
                            sub_pos = SetBasedPosition(sub_pos, self.n)
                            # add parent to dictionary
                            if sub_pos not in self.__complete_consistent_extensions.keys():
                                self.__complete_consistent_extensions[sub_pos] = set(self.__complete_consistent_extensions[position])
                                self.__consistent_parents[sub_pos] = {position}
                                self.__consistent_extensions[sub_pos] = {sub_pos}
                                self.__consistent_extensions[sub_pos].update(self.__consistent_extensions[position])
                            else:
                                self.__consistent_parents[sub_pos].add(position)
                                self.__consistent_extensions[sub_pos].update(self.__consistent_extensions[position])
                                self.__complete_consistent_extensions[sub_pos].update(
                                    set(self.__complete_consistent_extensions[position]))

                            new_gamma.add(sub_pos)
                        # reiterate again over layer and for each position:
                        # (i) calculate number of complete positions extending the position and
                        # (ii) determine its dialectical closure
                        for sub_pos in new_gamma:
                            self.__dict_n_complete_extensions[sub_pos] = len(self.__complete_consistent_extensions[sub_pos])
                            self.__n_extensions[sub_pos] = len(self.__consistent_extensions[sub_pos])
                            self.__closures[sub_pos] = sub_pos
                            for parent in self.__consistent_parents[sub_pos]:
                                if self.__dict_n_complete_extensions[sub_pos] == self.__dict_n_complete_extensions[parent]:
                                    self.__closures[sub_pos] = self.__closures[parent]
                                    break

                    current_gamma = new_gamma
                    new_gamma = set()
            # case: no arguments
            else:
                for pos in self.minimally_consistent_positions():
                    self.__closures[pos] = pos
                    self.__dict_n_complete_extensions[pos] = 2 ** (self.n - pos.size())
                    if pos.size() == self.n:
                        self.__n_extensions[pos] = 1
                    else:
                        self.__n_extensions[pos] = 1 + 2 ** (self.n - pos.size())
                    # Todo (to decrease ram-usage?)
                    # dealing with complete_consistent_extensions:
                    # instead of filling the dictionary, we deal with empty graphs in the function that would
                    # otherwise use complete_consistent_extensions
                    self.__complete_consistent_extensions[pos] = {pos}
                    # do we need them?
                    self.__consistent_parents = {}

            self.__dirty = False

    def add_argument(self, argument: List[int]):
        self.arguments.append(argument)
        self.cnf = self.__cnf(self.arguments)
        self.__dirty = True
        return self

    def add_arguments(self, arguments: List[List[int]]) -> DialecticalStructure:
        for arg in arguments:
            self.add_argument(arg)
        self.__dirty = True
        #self.__update()
        return self

    def get_arguments(self) -> List[List[int]]:
        return self.arguments

    def sentence_pool(self) -> Position:
        return SetBasedPosition(np.arange(1, self.n + 1), self.n)

    def __raise_value_error_if_sentence_pool_mismatch(self, position: Position):
        if position and self.sentence_pool() != position.sentence_pool():
            raise ValueError("The function you called expects positions to be based on the same sentence pool as",
                             " the dialectical structure.")

    def is_consistent(self, position: Position) -> bool:
        self.__raise_value_error_if_sentence_pool_mismatch(position)
        self.__update()
        if position.size() == 0:
            return True
        else:
            return SetBasedPosition.as_setbased_position(position) in self.__dict_n_complete_extensions.keys()

    def are_compatible(self, position1: Position, position2: Position) -> bool:
        self.__raise_value_error_if_sentence_pool_mismatch(position1)
        self.__raise_value_error_if_sentence_pool_mismatch(position2)

        if not self.is_consistent(position1) or not self.is_consistent(position2):
            return False
        if position1.size() == 0 or position2.size() == 0:
            return True
        if len(self.arguments) == 0:
            return (position1.union(position2)).is_minimally_consistent()
        else:
            return not self.__complete_consistent_extensions[SetBasedPosition.as_setbased_position(position2)].\
                isdisjoint(self.__complete_consistent_extensions[SetBasedPosition.as_setbased_position(position1)])

    def consistent_positions(self, position: Position = None) -> Iterator[Position]:
        self.__raise_value_error_if_sentence_pool_mismatch(position)
        if len(self.arguments) > 0:
            self.__update()
        if position is None:
            if len(self.arguments) == 0:
                return self.minimally_consistent_positions()
            else:
                return iter(self.__complete_consistent_extensions.keys())
        else:
            if len(self.arguments) == 0:
                return self.__it_minimally_consistent_positions(position)
            else:
                return self.__it_consistent_positions(position)

    def __it_minimally_consistent_positions(self, position: Position = None) -> Iterator[Position]:
        for pos in iter(self.minimally_consistent_positions()):
            if position.is_subposition(pos):
                yield pos

    def __it_consistent_positions(self, position: Position = None) -> Iterator[Position]:
        for pos in iter(self.__complete_consistent_extensions.keys()):
            if position.is_subposition(pos):
                yield pos

    def consistent_complete_positions(self, position: Position = None) -> Iterator[Position]:
        self.__raise_value_error_if_sentence_pool_mismatch(position)
        if len(self.arguments) > 0:
            self.__update()
        if position is None:
            if len(self.arguments) == 0:
                return self.complete_minimally_consistent_positions()
            else:
                return iter(self.complete_consistent_positions)
        else:
            if len(self.arguments) == 0:
                return self.__it_minimally_consistent_complete_positions(position)
            else:
                return self.__it_consistent_complete_positions(position)

    def __it_minimally_consistent_complete_positions(self, position: Position = None) -> Iterator[Position]:
        for pos in iter(self.complete_minimally_consistent_positions()):
            if position.is_subposition(pos):
                yield pos

    def __it_consistent_complete_positions(self, position: Position = None) -> Iterator[Position]:
        for pos in iter(self.complete_consistent_positions):
            if position.is_subposition(pos):
                yield pos

    def entails(self, position1: Position, position2: Position) -> bool:
        self.__raise_value_error_if_sentence_pool_mismatch(position1)
        self.__raise_value_error_if_sentence_pool_mismatch(position2)
        # contradiction entail everything (ex falso quodlibet)
        if not self.is_consistent(position1):
            return True
        # consistent positions do not entail inconsistent positions
        if not self.is_consistent(position2):
            return False
        # the empty position is entailed by everything
        if position2.size() == 0:
            return True
        # case of empty graph
        if len(self.arguments) == 0:
            return position1.intersection(position2) == position2
        else:
            return self.__complete_consistent_extensions[SetBasedPosition.as_setbased_position(position1)].\
                issubset(self.__complete_consistent_extensions[SetBasedPosition.as_setbased_position(position2)])

    def closure(self, position: Position) -> Position:
        self.__raise_value_error_if_sentence_pool_mismatch(position)
        if self.is_consistent(position):
            return self.__closures[SetBasedPosition.as_setbased_position(position)]
        # ex falso quodlibet
        else:
            return SetBasedPosition.from_set(set(self.__sentence_pool), self.n)

    def is_closed(self, position: Position) -> bool:
        self.__raise_value_error_if_sentence_pool_mismatch(position)
        return SetBasedPosition.as_setbased_position(position) == self.closure(position)

    def closed_positions(self) -> Iterator[Position]:
        self.__update()
        return iter({closure for closure in self.__closures.values()})

    def is_minimal(self, position: Position) -> bool:
        self.__raise_value_error_if_sentence_pool_mismatch(position)
        if not self.is_consistent(position):
            return True
        if position.size() == 0:
            return True
        # since we already checked for consistency, we only have to iterate over all subpositions (every subposition
        # of a consistent position will be consistent)
        for pos in (poss for poss in position.subpositions() if poss != position and poss.size() != 0):
            if self.entails(pos, position):
                return False

        return True

    def axioms(self, position: Position, source: Iterator[Position] = None) -> Iterator[Position]:
        self.__raise_value_error_if_sentence_pool_mismatch(position)
        if not self.is_consistent(position):
            raise ValueError('The given position is inconsistent.')
        if position.size() == 0:
            return {position}
        # if no source is provided, default to all consistent positions of dialectical structure
        if not source:
            source = self.consistent_positions()
        axioms = set()
        for pos in source:
            if self.entails(pos, position) and \
                    not any(self.entails(subpos, position) for subpos in pos.subpositions() if subpos != pos):
                axioms.add(pos)
        if len(axioms) == 0:
            return []
        return iter(axioms)

    def minimal_positions(self) -> Iterator[Position]:
        return iter(pos for pos in self.consistent_positions() if self.is_minimal(pos))

    def n_complete_extensions(self, position: Position = None) -> int:
        self.__update()
        if position is None or position.size() == 0:
        #    return len(self.complete_consistent_positions)
            return self.__dict_n_complete_extensions[SetBasedPosition({}, self.n)]
        if not self.is_consistent(position):
            return 0

        return self.__dict_n_complete_extensions[SetBasedPosition.as_setbased_position(position)]

    def is_complete(self, position: Position) -> bool:
        self.__raise_value_error_if_sentence_pool_mismatch(position)
        return position.domain().as_set() == set(self.__sentence_pool)

    def degree_of_justification(self, position1: Position, position2: Position) -> float:
        self.__raise_value_error_if_sentence_pool_mismatch(position1)
        self.__raise_value_error_if_sentence_pool_mismatch(position2)

        if not self.is_consistent(position2):
            raise(ZeroDivisionError())

        return self.n_complete_extensions(position1.union(position2))/self.n_complete_extensions(position2)

    def minimally_consistent_positions(self) -> Iterator[Position]:
        return self.__minimally_consistent_positions()

    def complete_minimally_consistent_positions(self) -> Iterator[Position]:
        return self.__minimally_consistent_positions(only_complete_positions=True)

    def __minimally_consistent_positions(self, propositions=None,
                                         only_complete_positions=False) -> Iterator[Position]:
        if propositions is None:
            propositions = list(np.arange(1, self.n + 1, 1)) + list(np.arange(-1, -self.n - 1, -1))
        if len(propositions) == 2:
            yield SetBasedPosition({propositions[0]}, self.n)
            yield SetBasedPosition({propositions[1]}, self.n)
            if not only_complete_positions:
                yield SetBasedPosition({}, self.n)
        else:
            for item in self.__minimally_consistent_positions(propositions[1:int(len(propositions) / 2)] +
                                                              propositions[int(len(propositions) / 2) +
                                                                           1:len(propositions)],
                                                              only_complete_positions):
                yield item.union(SetBasedPosition({propositions[0]}, self.n))
                yield item.union(SetBasedPosition({propositions[int(len(propositions) / 2)]}, self.n))
                if not only_complete_positions:
                    yield SetBasedPosition(item, self.n)

    # def __minimally_consistent_positions2(self, propositions = None,
    #                                      only_complete_positions = False) -> Iterator[Position]:
    #     if propositions is None:
    #         propositions = list(np.arange(1, self.n + 1, 1)) +  list(np.arange(-1, -self.n - 1, -1))
    #     if len(propositions) == 2:
    #         yield {propositions[0]}
    #         yield {propositions[1]}
    #         if not only_complete_positions:
    #             yield set()
    #     else:
    #         for item in self.__minimally_consistent_positions(propositions[1:int(len(propositions) / 2)] +
    #                                                           propositions[int(len(propositions) / 2) +
    #                                                                        1:len(propositions)],
    #                                                           only_complete_positions):
    #             yield SetBasedPosition({propositions[0]}.union(item), self.n)
    #             yield SetBasedPosition({propositions[int(len(propositions) / 2)]}.union(item), self.n)
    #             if not only_complete_positions:
    #                 yield SetBasedPosition(item, self.n)

