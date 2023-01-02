# Todo (@Andreas): Add module docstring.

from .base import Position
from .base import DialecticalStructure
from .numpy_implementation import NumpyPosition

from bitarray import bitarray
from typing import Set, Iterator, List, Union
from itertools import combinations, product

import logging
logging.basicConfig(filename='bitarray_implementation.log', level=logging.INFO)


class BitarrayPosition(Position):
    """using bitarrays to represent positions and their methods

    A pair of bits represents a sentence:
    The first bit represents acceptance and the second bit rejection
    Suspension of a sentence corresponds to both bits being False/0
    (Minimal or flat) contradiction is present if both bits are True/1

    A BitarrayPosition may be created by a bitarray,
    e.g. BitarrayPosition(bitarray('10001001')), or by a set
    of integer-represented sentences and the number of unnegated sentences in the
    sentencepool, e.g. BitarrayPosition({1, 3, -4}, n_unnegated_sentence_pool = 4))
    """

    def __init__(self, ba: Union[bitarray, Set[int]], n_unnegated_sentence_pool: int = None):
        """initialized by a bitarray of a string of zeros and ones"""
        if type(ba) == bitarray:
            self.__bitarray = ba
            self.n_unnegated_sentence_pool = int(len(ba)/2)

        else:   # convert set into bitarray
            if n_unnegated_sentence_pool:
                position_ba = 2 * n_unnegated_sentence_pool * bitarray('0')
                self.n_unnegated_sentence_pool = n_unnegated_sentence_pool
            else:
                raise ValueError("Initialization with set requires specified n_unnegated_sentence_pool.")

            for sentence in ba:
                if sentence < 0:
                    position_ba[2 * (abs(sentence) - 1) + 1] = True
                else:
                    position_ba[2 * (abs(sentence) - 1)] = True
            self.__bitarray = position_ba
        super().__init__()


    # representation as string when a Position is printed, for example
    def __repr__(self) -> str:
        # return bitarray.to01(self.__bitarray)
        return str(self.as_set())

    # Remark: Apparently, if `__eq__` is implemented, you have to implement `__hash__`
    # explicitly in the same class (i.e., it is not enough to have `__hash__` implemented in the
    # super class).
    def __hash__(self):
        #return hash(self.__bitarray.to01())
        return super().__hash__()

    def __eq__(self, other) -> bool:
        if isinstance(other, Position):
            return self.as_bitarray() == other.as_bitarray()
        else:
            return False


    @staticmethod
    def from_set(position: Set[int], n_unnegated_sentence_pool: int) -> Position:
        position_ba = 2 * n_unnegated_sentence_pool * bitarray('0')
        for sentence in position:
            if sentence < 0:
                position_ba[2 * (abs(sentence) - 1) + 1] = True
            else:
                position_ba[2 * (abs(sentence) - 1)] = True
        return BitarrayPosition(position_ba, n_unnegated_sentence_pool)

    def sentence_pool(self) -> int:
        return self.n_unnegated_sentence_pool

    # number of a position's non-supended sentences
    def size(self) -> int:
         return self.__bitarray.count()

    def domain(self) -> Position:
        dom = ''
        for i in range(int(len(self.__bitarray) / 2)):
            if self.__bitarray[2 * i:2 * i + 2].any():
                dom += '11'
            else:
                dom += '00'
        return BitarrayPosition(bitarray(dom))

    # methods that return positions in different ways

    def as_bitarray(self) -> bitarray:
        return self.__bitarray

    def as_ternary(self) -> int:
        if self.is_minimally_consistent():

            # 0:suspension, 2:rejection, 1:acceptance
            a = sum(1 * 10 ** index for index, val in enumerate(self.__bitarray[0::2]) if val)
            r = sum(2 * 10 ** index for index, val in enumerate(self.__bitarray[1::2]) if val)
            return a + r
        else:
            return None

    def as_set(self) -> Set[int]:
        return set(self.as_list())

    def as_list(self) -> List[int]:
        res = []
        positive = self.__bitarray[0::2]
        negative = self.__bitarray[1::2]
        for i in range(int(len(self.__bitarray) / 2)):
            if positive[i]:
                res.append(i+1)
            if negative[i]:
                res.append(-(i+1))
        return res

    def is_minimally_consistent(self) -> bool:
        return not any(self.__bitarray[0::2] & self.__bitarray[1::2])

    def are_minimally_compatible(self, position1: Position) -> bool:
        return self.union([self, position1]).is_minimally_consistent()

    def is_subposition(self: Position, pos2: Position) -> bool:
        try:
            self.as_bitarray() ^ pos2.as_bitarray()
        except ValueError:
            logging.error("is supposition: bitwise comparison of " + str(self.as_bitarray()) + " and "
                                  + str(pos2.as_bitarray()) + " failed due to different sizes.")
            raise
        else:
            return not any(self.as_bitarray() & (self.as_bitarray() ^ pos2.as_bitarray()))

    """
    def subpositions(self, n: int = -1, only_consistent_subpositions: bool = True) -> Iterator[Position]:

        if n == -1:
            # by default, create set of all subpositions:
            res = set()
            for i in range(len(self.__bitarray) + 1):
                res.update(self.subpositions(i, only_consistent_subpositions))

            return iter(res)

        else:
            # all masks for subpositions
            if only_consistent_subpositions and self.is_minimally_consistent():
                submasks = [bitarray(''.join(e)) for e in product(['00', '11'], repeat=int(len(self.__bitarray) / 2))]
            else:
                submasks = [bitarray(''.join(e)) for e in product(['0', '1'], repeat=int(len(self.__bitarray)))]

            subpositions = set()
            for subm in submasks:
                ba = self.__bitarray & subm  # values of new bitarray
                # count non-suspended sentences and exclude positions that are too small
                if ba.count() == n:
                    new_supposition = BitarrayPosition(ba)
                    if only_consistent_subpositions and new_supposition.is_minimally_consistent():
                        subpositions.add(BitarrayPosition(ba))
                    elif not only_consistent_subpositions:
                        subpositions.add(BitarrayPosition(ba))
            return iter(subpositions)
    """

    def subpositions(self, n: int = -1, only_consistent_subpositions: bool = True) -> Iterator[Position]:
        """ Returns an iterator over subpositions of size n, with n being an optional
        argument (defaulting to -1, returning all subpositions.
        If only_consistent_subposition is set to true (by default), only minimally
        consistent subpositions are returned, which is less costly than returning all subpositions
        (if the parameter is set to false)).
        """

        if n == -1:
            # by default, create set of all subpositions:
            res = set()
            for i in range(len(self.__bitarray) + 1):
                res.update(self.subpositions(i, only_consistent_subpositions))

            return iter(res)

        else:
            subpositions = [BitarrayPosition(pos, n_unnegated_sentence_pool=int(len(self.__bitarray)/2))
                            for pos in combinations(self.as_set(), n)]

            if only_consistent_subpositions and not self.is_minimally_consistent():
                subpositions = [position for position in subpositions if position.is_minimally_consistent()]

            return iter(subpositions)

    def direct_subpositions(self) -> Iterator[Position]:
        """Creates subpositions of position that have exactly one element less than postion.
        Used for efficiency in update method."""
        # ToDo: Move to update method? Make it private?
        # ToDo: To be used inside method `subpositions` for more efficiency?
        # ToDo: catch minimally inconsistent positions!
        res = []
        # 1/2 number of bits in bitarray
        n = int(len(self.__bitarray)/2)
        # number of sentences in position
        s = self.size()
        for submask in [bitarray('11'*i + '00' + '11'*(n-1-i)) for i in range(n)]:
            subpos = submask & self.__bitarray
            if subpos.count() == s-1:
                res.append(BitarrayPosition(subpos))

        return res

    def is_accepting(self, sentence: int) -> bool:

        if not self.size():   # self is the empty position
            return False

        if sentence < 0:    # rejected sentence
            s = abs(sentence)-1
            return self.__bitarray[2 * s + 1]

        else:   # accepted sentence
            s = sentence-1
            return self.__bitarray[2 * s]

    def is_in_domain(self, sentence: int) -> bool:
        return any(self.__bitarray[2 * (abs(sentence) - 1):2 * (abs(sentence) - 1) + 2])

    @staticmethod
    def union(positions: Set[Position]) -> Position:
        if not positions:
            return BitarrayPosition.from_set(set(), 0)
        if len(positions) == 1:
            return next(iter(positions))
        else:
            position_list = list(positions)
            ba = position_list.pop().as_bitarray().copy()
            # create union with bitwise OR: |
            for pos in position_list:
                try:
                    ba |= pos.as_bitarray()
                except ValueError:
                    logging.error("Union: bitwise comparison of " + str(ba) + " and "
                                  + str(pos.as_bitarray()) + " failed due to different sizes.")

                    # reraise error to interrupt program
                    raise

        return BitarrayPosition(ba)

    @staticmethod
    def intersection(positions: Set[Position]) -> Position:

        # ToDo: discuss intersection of empty sets
        # bad fix for empty set of positions
        if not positions:
            return BitarrayPosition(bitarray('0'))

        elif len(positions) == 1:
            return BitarrayPosition(next(iter(positions)).as_bitarray())
        else:
            position_list = [BitarrayPosition(pos.as_bitarray()) for pos in positions]
            ba = position_list.pop().as_bitarray().copy()
            # create intersection with bitwise AND: &
            for pos in position_list:
                try:
                    ba &= pos.as_bitarray()
                except ValueError:
                    logging.error("Intersection: bitwise comparison of " + str(ba) + " and "
                                  + str(pos.as_bitarray()) + " failed due to different sizes.")

                    # reraise error to interrupt program
                    raise

        return BitarrayPosition(ba)

    def neighbours(self, depth: int) -> Iterator[Position]:
        for neighbour in NumpyPosition.np_neighbours(self, depth):
            yield BitarrayPosition.from_set(NumpyPosition(neighbour).as_set(), self.sentence_pool())

# Todo (@Andreas): Add class docstring.
class DAGBitarrayDialecticalStructure(DialecticalStructure):

    def __init__(self, n: int, initial_arguments: List[List[int]] = None):
        self.arguments_cnf = set()
        self.arguments = []
        self.n = n      # number of unnegated sentences in sentence pool used to iterate through positions
        self.__sp = BitarrayPosition(bitarray('1') * 2 * n)   # full sentence pool

        # prepare storage for results of costly functions
        # ToDo: Which of the following may be made private?
        self.cons_comp_pos = set()
        self.consistent_parents = {}
        self.complete_consistent_extensions = {}
        self.closures = {}
        self.min_cons_pos = set()

        ###
        self.consistent_extensions = {}
        self.dict_n_complete_extensions = {}
        self.n_extensions = {}

        # update status
        self.n_updates = 0
        self.__updated = False

        if initial_arguments:
            for arg in initial_arguments:
                self.add_argument(arg)

            # update the dialectical structure
            self.__update()

    @staticmethod
    def from_arguments(arguments: List[List[int]], n_unnegated_sentence_pool: int) -> DialecticalStructure:
        return DAGBitarrayDialecticalStructure(n_unnegated_sentence_pool, arguments)

    def to_bitarray_position(self, position: Position) -> BitarrayPosition:
        """This auxiliary method converts a position from another implemenation to a BitarrayPosition."""
        if isinstance(position, BitarrayPosition):
            return position
        elif position == bitarray():   # catch empty position
            return BitarrayPosition(2 * self.n * bitarray('0'))
        else:
            return BitarrayPosition(position.as_bitarray())

    def add_argument(self, argument: List[int]) -> DialecticalStructure:
        # arguments are represented as a single position (of its negated premises + its conclusion)
        # as it stands, the original arguments are not retrievable from this representation

        arg = bitarray('0')*2*self.n

        # negated premises
        for i in argument[:-1]:
            if i < 0:
                arg[2 * (abs(i) - 1)] = True
            else:
                arg[2 * (abs(i) - 1) + 1] = True

        # conclusion
        j = argument[-1]
        if j < 0:
            arg[2 * (abs(j) - 1) + 1] = True
        else:
            arg[2 * (j - 1)] = True

        self.arguments_cnf.add(BitarrayPosition(arg))
        self.arguments.append(argument)
        # If new arguments are added to a dialectical structure, already existing
        # objects such as the DAG or the closures may need to be recreated. The need
        # for an update is indicated by setting its update status to False.
        self.__updated = False

        return self

    def add_arguments(self, arguments: List[List[int]]) -> DialecticalStructure:
        for argument in arguments:
            self.add_argument(argument)
        self.__update()
        return self

    def get_arguments(self) -> List[List[int]]:
        return self.arguments

    '''
    Sentence-pool, domain and completeness
    '''

    def is_complete(self, position: Position) -> bool:
        return self.__sp == position.domain()

    def sentence_pool(self) -> Position:
        """Returns the unnegated half of sentence pool"""

        return BitarrayPosition(bitarray('10')*self.n)

    '''
    Dialectic consistency & compatibility
    '''

    def satisfies(self, argument: Position, position: Position) -> bool:
        try:
            return any(argument.as_bitarray() & position.as_bitarray())
        except ValueError:
            logging.error("satisfies: bitwise comparison of " + str(argument)
                          + ", " + str(position) + " and " + " failed due to different sizes.")
            raise

    def is_consistent(self, position: Position) -> bool:
        # check update status of dialectical structure
        self.__update()

        # position is converted to a BitarrayPosition to ensure compatability with other implementations

        return self.to_bitarray_position(position) in self.complete_consistent_extensions.keys()

    def are_compatible(self, position1: Position, position2: Position, ) -> bool:
        # check update status of dialectical structure
        self.__update()

        # if necessary, convert to BitarrayPosition
        position1 = self.to_bitarray_position(position1)
        position2 = self.to_bitarray_position(position2)

        # case: at least one of the positions is not minimally consistent
        if not (position1.is_minimally_consistent() and position2.is_minimally_consistent()):
            return False
        # case: both positions are minimally consistent
        else:
            return any(self.complete_consistent_extensions[position1].intersection(self.complete_consistent_extensions[position2]))

    def minimally_consistent_positions(self) -> Iterator[Position]:
        # auxiliary function to convert a number from base 10 to
        # base 3 (reversed) and then to a position
        def number_to_position(number):
            # convert to ternary
            if number == 0:
                return '0'
            nums = []
            while number:
                number, r = divmod(number, 3)
                nums.append(str(r))
            ter = ''.join(nums)
            if len(ter) < self.n:
                ter = ter + ''.join(['0' for _ in range(self.n - len(ter))])
            res = ''
            # convert ternary to bitarray string
            for b in ter:
                # suspension
                if b == '0':
                    res += '00'
                # acceptance
                elif b == '1':
                    res += '10'
                # rejection
                else:
                    res += '01'

            return BitarrayPosition(bitarray(res))

        # create the set of all positions excluding empty position
        min_consistent_positions = set()
        for i in range(1, 3 ** self.n):
            min_consistent_positions.add(number_to_position(i))

        # add empty position
        min_consistent_positions.add(BitarrayPosition(bitarray('0'*2*self.n)))

        self.min_cons_pos = min_consistent_positions
        return iter(min_consistent_positions)

    def consistent_positions(self) -> Iterator[Position]:
        # check update status of dialectical structure
        self.__update()

        # note that the complete_parent_graph and direct_parent_graph have the
        # same keys
        return iter(self.complete_consistent_extensions.keys())

    def consistent_complete_positions(self) -> Iterator[Position]:
        # self.__update()
        if not self.cons_comp_pos:
            all_complete_positions = [BitarrayPosition(bitarray(''.join(e)))
                                      for e in product(['10', '01'], repeat=self.n)]

            # filter positions that do not satisfy all arguments:
            self.cons_comp_pos = set(
                    [pos for pos in all_complete_positions
                     if all(self.satisfies(arg, pos) for arg in self.arguments_cnf)])

            return iter(self.cons_comp_pos)

        else:
            # check update status of dialectical structure
            # self.check_update()

            return iter(self.cons_comp_pos)

    '''
    Dialectic entailment and dialectic closure of consistent positions
    '''

    def entails(self, position1: Position, position2: Position) -> bool:
        # check update status of dialectical structure
        self.__update()

        # if necessary, convert to BitarrayPosition
        position1 = self.to_bitarray_position(position1)
        position2 = self.to_bitarray_position(position2)

        if not self.is_consistent(position1):     # ex falso quodlibet
            return True
        elif not self.is_consistent(position2):
            return False

        # catch complete positions (else-statements)
        if self.complete_consistent_extensions[position1]:
            pos1_extensions = self.complete_extensions(position1)
        else:
            pos1_extensions = {position1}

        if self.complete_consistent_extensions[position2]:
            pos2_extensions = self.complete_extensions(position2)
        else:
            pos2_extensions = {position2}

        return pos1_extensions.issubset(pos2_extensions)

    def closure(self, position: Position) -> Position:
        if self.is_consistent(position):
            if position.size() == 0:
                return BitarrayPosition.intersection(self.cons_comp_pos)
            else:
                return self.closures[self.to_bitarray_position(position)]
        # ex falso quodlibet
        else:
            return self.__sp

    def is_closed(self, position: Position) -> bool:
        # check update status of dialectical structure
        self.__update()

        return self.to_bitarray_position(position) == self.closure(self.to_bitarray_position(position))

    def closed_positions(self) -> Iterator[Position]:
        self.__update()
        return iter([pos for pos in self.consistent_positions() if self.is_closed(pos)])

    def axioms(self, position: Position, source: Iterator[Position] = None) -> Iterator[Position]:
        if not self.is_consistent(position):
            logging.error(position)
            logging.error(self.complete_consistent_extensions)
            logging.error(self.n_updates)
            raise ValueError("An inconsistent Position cannot be axiomatized!")

        res = set()

        # if no source is provided, default to all consistent positions of dialectical structure
        if not source:
            source = self.consistent_positions()

        #  collect minimal positions from *source*, which entail *position*
        for pos in source:
            if (self.entails(pos, position)
                    and not any(self.entails(subpos, position) for subpos in pos.subpositions() if subpos != pos)):

                res.add(pos)

        if not res:
            return None

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
    def complete_extensions(self, position: Position) -> Set[Position]:
        """Returns complete extensions of position by retrieving corresponding
         node in the graph that stores complete extensions."""

        # check update status of dialectical structure
        self.__update()

        if not position.is_minimally_consistent():
            return set()
        # complete positions have no parents but extend themselves
        elif len(self.complete_consistent_extensions[self.to_bitarray_position(position)]) == 0:
            return {position}
        else:
            return self.complete_consistent_extensions[self.to_bitarray_position(position)]

    # sigma
    def n_complete_extensions(self, position: Position = None) -> int:
        self.__update()
        if not position or position.size() == 0:
            return len(self.cons_comp_pos)
        else:
            return len(self.complete_extensions(position))

    # conditional doj
    def degree_of_justification(self, position1: Position, position2: Position) -> float:

        return len(self.complete_extensions(position1).intersection(
            self.complete_extensions(position2)))/self.n_complete_extensions(position2)

    def __update(self) -> None:
        """Checks whether the dialectical structure is up to date,
        which may not be the case after new arguments have been added after its
        initialisation.
        If the structure is outdated, this method resets and recreates stored
        objects (e.g. DAG, closure).
        All methods that require an updated structure to work properly, call this method."""

        if not self.__updated:
            # ToDo: Which ones are really important to keep and which ones can be set private?
            self.complete_consistent_extensions = {}
            self.consistent_extensions = {}
            self.consistent_parents = {}

            self.dict_n_complete_extensions = {}
            self.n_extensions = {}
            self.closures = {}

            # Minimally consistency is not affected by structural updates,
            # but it is created if it does not already exist.
            if not self.min_cons_pos:
                self.minimally_consistent_positions()

            # starting point (gamma_n): consistent complete positions
            current_gamma = set(self.consistent_complete_positions())

            for pos in current_gamma:
                # complete positions have no parents
                self.complete_consistent_extensions[pos] = {pos}
                self.consistent_parents[pos] = set()
                self.consistent_extensions[pos] = {pos}
                self.dict_n_complete_extensions[pos] = 1
                self.closures[pos] = pos
                self.n_extensions[pos] = 1

            new_gamma = set()
            # iterate backwards over length of subpositions
            for i in reversed(range(0, self.n)):
                for position in current_gamma:
                    #for sub_pos in position.subpositions(i, only_consistent_subpositions=True):
                    for sub_pos in position.direct_subpositions():

                        # add parent to dictionary
                        if sub_pos not in self.complete_consistent_extensions.keys():
                            self.complete_consistent_extensions[sub_pos] = set(self.complete_consistent_extensions[position])
                            self.consistent_parents[sub_pos] = {position}
                            self.consistent_extensions[sub_pos] = {sub_pos}
                            self.consistent_extensions[sub_pos].update(self.consistent_extensions[position])
                        else:
                            self.consistent_parents[sub_pos].add(position)
                            self.consistent_extensions[sub_pos].update(self.consistent_extensions[position])
                            self.complete_consistent_extensions[sub_pos].update(
                                set(self.complete_consistent_extensions[position]))

                        new_gamma.add(sub_pos)

                for pos in new_gamma:

                    # Gregor's Condition

                    if any(len(self.complete_consistent_extensions[pos]) == len(self.complete_consistent_extensions[par])
                            for par in self.consistent_parents[pos]):

                        self.closures[pos] = self.closures[next(par for par in self.consistent_parents[pos]
                                                    if len(self.complete_consistent_extensions[pos]) == len(self.complete_consistent_extensions[par]))]

                    else:
                        self.closures[pos] = pos

                current_gamma = new_gamma
                new_gamma = set()

        # update status of dialectical structure
        self.__updated = True

        return None
