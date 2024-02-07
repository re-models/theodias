# Todo (@Basti): Add module docstring.

# see: https://stackoverflow.com/questions/33533148
from __future__ import annotations

from .base import Position, DialecticalStructure
from .numpy_implementation import NumpyPosition

from bitarray import bitarray
from typing import Set, Iterator, List
from itertools import chain, combinations
from pysat.formula import CNF
from pysat.solvers import Minisat22
import numpy as np

# Todo (@Basti): Add class docstring.
class SetBasedPosition(Position):

    # comment: `def __init__(self, position1: Set[int] = set()):` should do as well, but it doesn't... beats me
    def __init__(self, position: Set[int], n_unnegated_sentence_pool: int):
        self.__position = frozenset(position)
        self.n_unnegated_sentence_pool = n_unnegated_sentence_pool
        super().__init__()


    # representation as string when a Position is printed, for example
    def __repr__(self) -> str:
        # ToDo: rewrite without relying on assumptions about what frozenset.__repr__ returns!
        #return self.__position.__repr__().replace('frozenset','SetBasedPosition')
        return set(self.__position).__repr__()
    # hashing enables to form sets of Positions
    #def __hash__(self):
    #    return self.__position.__hash__()

    def __iter__(self):
        return self.__position.__iter__()

    def __next__(self):
        return self.__position.__next__()

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
        if self.is_minimally_consistent():
            res = 0
            for sentence in self.__position:
                if sentence < 0:
                    res += 2 * 10 ** (abs(sentence) - 1)
                else:
                    res += 1 * 10 ** (sentence - 1)
            return res
        else:
            return None

    def as_set(self) -> Set[int]:
        return set(self.__position)

    def as_list(self) -> List[int]:
        return list(self.__position)

    def is_minimally_consistent(self) -> bool:
        return not any([-1 * element in self.__position for element in self.__position])

    def are_minimally_compatible(self, position: Position) -> bool:
        return SetBasedPosition(self.as_set() | position.as_set(),
                                max(self.sentence_pool().size(),
                                    position.sentence_pool().size())).is_minimally_consistent()

    def is_subposition(self, position: Position) -> bool:
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

    @staticmethod
    def union(positions: Set[Position]) -> Position:
        if not positions:
            return SetBasedPosition(set(),0)
        n_sentence_pool = max([pos.sentence_pool().size() for pos in positions])
        ret = set()
        for position in positions:
            ret = ret | position.as_set()
        return SetBasedPosition(list(ret), n_sentence_pool)

    @staticmethod
    def intersection(positions: Set[Position]) -> Position:
        if not positions:
            return SetBasedPosition(set(),0)
        n_sentence_pool = max([pos.sentence_pool().size() for pos in positions])
        ret = positions.pop().as_set()
        for position in positions:
            ret = ret & position.as_set()
        return SetBasedPosition(list(ret), n_sentence_pool)

    def difference(self, pos: Position ) -> Position:
        return SetBasedPosition(self.as_set().difference(pos.as_set()), pos.sentence_pool().size())

    def neighbours(self, depth: int) -> Iterator[Position]:
        for neighbour in NumpyPosition.np_neighbours(self, depth):
            yield SetBasedPosition.from_set(NumpyPosition(neighbour).as_set(), self.sentence_pool().size())

# Todo (@Basti): Add class docstring.
class DAGSetBasedDialecticalStructure(DialecticalStructure):

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

    # todo: consider empty ds
    def __update(self):
        if self.__dirty:
            # ToDo: Which ones are really important to keep and which ones can be set private?
            self.complete_consistent_extensions = {}
            # These two I call only in the constructor
            self.__consistent_extensions = {}
            self.__consistent_parents = {}

            self.dict_n_complete_extensions = {}
            self.n_extensions = {}
            self.closures = {}

            if self.arguments:

                self.complete_consistent_positions = self.compute_complete_consistent_positions()
                current_gamma = self.complete_consistent_positions
                for pos in current_gamma:
                    # complete positions have no parents
                    self.complete_consistent_extensions[pos] = {pos}
                    self.__consistent_extensions[pos] = {pos}
                    self.dict_n_complete_extensions[pos] = 1
                    self.closures[pos] = pos
                    self.n_extensions[pos] = 1

                new_gamma = set()
                # iterate backwards over length of subpositions
                for i in reversed(range(0, self.n)):
                    for position in current_gamma:
                        for sub_pos in combinations(position, i):
                            sub_pos = SetBasedPosition(sub_pos, self.n)
                            # add parent to dictionary
                            if sub_pos not in self.complete_consistent_extensions.keys():
                                self.complete_consistent_extensions[sub_pos] = set(self.complete_consistent_extensions[position])
                                self.__consistent_parents[sub_pos] = {position}
                                self.__consistent_extensions[sub_pos] = {sub_pos}
                                self.__consistent_extensions[sub_pos].update(self.__consistent_extensions[position])
                            else:
                                self.__consistent_parents[sub_pos].add(position)
                                self.__consistent_extensions[sub_pos].update(self.__consistent_extensions[position])
                                self.complete_consistent_extensions[sub_pos].update(
                                    set(self.complete_consistent_extensions[position]))

                            new_gamma.add(sub_pos)
                        # reiterate again over layer and for each position:
                        # (i) calculate number of complete positions extending the position and
                        # (ii) determine its dialectical closure
                        for sub_pos in new_gamma:
                            self.dict_n_complete_extensions[sub_pos] = len(self.complete_consistent_extensions[sub_pos])
                            self.n_extensions[sub_pos] = len(self.__consistent_extensions[sub_pos])
                            self.closures[sub_pos] = sub_pos
                            for parent in self.__consistent_parents[sub_pos]:
                                if self.dict_n_complete_extensions[sub_pos] == self.dict_n_complete_extensions[parent]:
                                    self.closures[sub_pos] = self.closures[parent]
                                    break

                    current_gamma = new_gamma
                    new_gamma = set()
            # case: no arguments
            else:
                # dealing with complete_consistent_extensions:
                # instead of filling the dictionary, we deal with empty graphs in the function that would
                # otherwise use complete_consistent_extensions
                for pos in self.minimally_consistent_positions():
                    self.closures[pos] = pos
                    self.dict_n_complete_extensions[pos] = 2**(self.n - pos.size())
                    if pos.size() == self.n:
                        self.n_extensions[pos] = 1
                    else:
                        self.n_extensions[pos] = 1 + 2**(self.n - pos.size())
                    # todo
                    self.complete_consistent_extensions[pos] = {pos}

                    # do we need them
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

    # ToDo: How do we deal with positions that indicate a larger sentencepool?
    # Either throwing an error or leave it as it is (treating them as if the are inconsistent).
    # The first one what I would expect from the user point of view, but it is possibly costly.
    def is_consistent(self, position: Position) -> bool:
        self.__update()
        if position.size() == 0:
            return True
        else:
            return SetBasedPosition.as_setbased_position(position) in self.dict_n_complete_extensions.keys()

    def are_compatible(self, position1: Position, position2: Position) -> bool:
        if not self.is_consistent(position1) or not self.is_consistent(position2):
            return False
        if position1.size() == 0 or position2.size() == 0:
            return True
        if len(self.arguments) == 0:
            return SetBasedPosition.union({position1, position2}).is_minimally_consistent()
        else:
            return not self.complete_consistent_extensions[SetBasedPosition.as_setbased_position(position2)].\
                isdisjoint(self.complete_consistent_extensions[SetBasedPosition.as_setbased_position(position1)])

    def consistent_positions(self) -> Iterator[Position]:
        self.__update()
        if len(self.arguments) == 0:
            return self.minimally_consistent_positions()
        return iter(self.complete_consistent_extensions.keys())

    def consistent_complete_positions(self) -> Iterator[Position]:
        self.__update()
        if len(self.arguments) == 0:
            return self.complete_minimally_consistent_positions()
        return iter(self.complete_consistent_positions)

    def entails(self, position1: Position, position2: Position) -> bool:
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
            return SetBasedPosition.intersection({position1, position2}).as_set() == position2.as_set()
        else:
            return self.complete_consistent_extensions[SetBasedPosition.as_setbased_position(position1)].\
                issubset(self.complete_consistent_extensions[SetBasedPosition.as_setbased_position(position2)])

    def closure(self, position: Position) -> Position:
        if self.is_consistent(position):
            return self.closures[SetBasedPosition.as_setbased_position(position)]
        # ex falso quodlibet
        else:
            return SetBasedPosition.from_set(set(self.__sentence_pool), self.n)

    def is_closed(self, position: Position) -> bool:
        return SetBasedPosition.as_setbased_position(position) == self.closure(position)

    def closed_positions(self) -> Iterator[Position]:
        self.__update()
        return iter({closure for closure in self.closures.values()})


    def is_minimal(self, position: Position) -> bool:
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
        if not axioms:
            return []
        return iter(axioms)

    def minimal_positions(self) -> Iterator[Position]:
        return iter(pos for pos in self.consistent_positions() if self.is_minimal(pos))

    def n_complete_extensions(self, position: Position = None) -> int:
        self.__update()
        if position is None or position.size() == 0:
        #    return len(self.complete_consistent_positions)
            return self.dict_n_complete_extensions[SetBasedPosition({},self.n)]
        if not self.is_consistent(position):
            return 0

        return self.dict_n_complete_extensions[SetBasedPosition.as_setbased_position(position)]

    def is_complete(self, position: Position) -> bool:
        return position.domain().as_set() == set(self.__sentence_pool)

    # ToDo: Is the cut of the complete consistent extensions of two positions A and B the same as set
    # of complete consistent extension of $A\cup B$?
    def degree_of_justification(self, position1: Position, position2: Position) -> float:
        if not self.is_consistent(position2):
            raise(ZeroDivisionError())

        return self.n_complete_extensions(SetBasedPosition.union([position1, position2]))/\
               self.n_complete_extensions(position2)

    def minimally_consistent_positions(self) -> Iterator[Position]:
        return self.__minimally_consistent_positions()

    def complete_minimally_consistent_positions(self) -> Iterator[Position]:
        return self.__minimally_consistent_positions(only_complete_positions = True)

    def __minimally_consistent_positions(self, propositions = None,
                                         only_complete_positions = False) -> Iterator[Position]:
        if propositions is None:
            propositions = list(np.arange(1, self.n + 1, 1)) +  list(np.arange(-1, -self.n - 1, -1))
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
                yield SetBasedPosition.union({SetBasedPosition({propositions[0]}, self.n), item})
                yield SetBasedPosition.union({SetBasedPosition({propositions[int(len(propositions) / 2)]}, self.n),
                                              item})
                if not only_complete_positions:
                    yield SetBasedPosition(item, self.n)

    def __minimally_consistent_positions2(self, propositions = None,
                                         only_complete_positions = False) -> Iterator[Position]:
        if propositions is None:
            propositions = list(np.arange(1, self.n + 1, 1)) +  list(np.arange(-1, -self.n - 1, -1))
        if len(propositions) == 2:
            yield {propositions[0]}
            yield {propositions[1]}
            if not only_complete_positions:
                yield set()
        else:
            for item in self.__minimally_consistent_positions(propositions[1:int(len(propositions) / 2)] +
                                                              propositions[int(len(propositions) / 2) +
                                                                           1:len(propositions)],
                                                              only_complete_positions):
                yield SetBasedPosition({propositions[0]}.union(item), self.n)
                yield SetBasedPosition({propositions[int(len(propositions) / 2)]}.union(item), self.n)
                if not only_complete_positions:
                    yield SetBasedPosition(item, self.n)

