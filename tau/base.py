"""
.. module:: base
    :synopsis: module defining basic abstract classes

"""

# see: https://stackoverflow.com/questions/33533148
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Set, Iterator, List, TypedDict, final
import numpy as np
from bitarray import bitarray



class Position(ABC):
    """Class representing a position.

        A position :math:`\\mathcal{A}` is a subset :math:`\\mathcal{A}\\subset S` over a sentence pool
        :math:`S = \\{ s_1, s_2, \\dots, s_N, \\neg s_1, \\neg s_2, \\dots, \\neg s_N \\}`.

        .. note::

            Implementations of this abstract class do not necessarily adhere to a set-theoretic representation.
    """

    # deprecated: Position should be imutable
    # @abstractmethod
    # def add(self, sentence: int) -> Position:
    #     """ Adding the acceptance of a sentence.
    #
    #     :param sentence: The sentence as its int-representation.
    #     :return: The position for convenience.
    #     """
    #     pass
    #
    # @abstractmethod
    # def remove(self, sentence: int) -> Position:
    #     """ Removing the acceptance of a sentence.
    #
    #     :param sentence: The sentence as its int-representation.
    #     :return: The position for convenience.
    #     """
    #     pass

    def __init__(self):
        self.__hash = None

    # Two positions are equal to each other, if they are based on the same sentence pool and if they represent
    # the same set of sentences. Subclasses should override this function only for speed of computation. They should
    # not change the behaviour of this function.
    def __eq__(self, other):
        if isinstance(other, Position):
            return self.as_set() == other.as_set() and self.sentence_pool().size() == other.sentence_pool().size()
        else:
            return False

    # The hash key should be the same for different implementations. It should depend on the position and
    # the sentencepool.
    def __hash__(self):
        if not self.__hash:
            # we take as base 5 in order to specify the sentencepool in the int-representation
            # 1 represents suspension, 2 belief, 3 disbelief, 4 belief and disbelief
            arr = np.ones(self.sentence_pool().size(), dtype=np.float32)
            for s in self.as_set():
                if s < 0:
                    arr[abs(s) - 1] += 2
                elif s > 0:
                    arr[s - 1] += 1
            s = 0
            for i in range(len(arr)):
                s += arr[i] * 5 ** i
            self.__hash = hash(s)

        return self.__hash

    @abstractmethod
    def sentence_pool(self) -> Position:
        """Returns the sentences (without negations) of a position's sentence pool.

        Returns from :math:`S = \\{ s_1, s_2, \\dots, s_N, \\neg s_1, \\neg s_2, \\dots, \\neg s_N \\}`
        only :math:`\\{ s_1, s_2, \\dots, s_N\\}`

        :return: "Half" of the full sentence pool :math:`S` as :code:`Position` (sentences without their negation).
        """
        pass

    @abstractmethod
    def domain(self) -> Position:
        """Determines the domain of the position.

        The domain of a position :math:`\\mathcal{A}` is the closure of :math:`\\mathcal{A}` under negation.

        :returns: the domain of the position
        """
        pass

    @staticmethod
    @abstractmethod
    def from_set(position: Set[int], n_unnegated_sentence_pool: int) -> Position:
        """Instanciating a :class:`Position` from a set.

        :return: :class:`Position`
        """
        pass

    @abstractmethod
    def as_bitarray(self) -> bitarray:
        """Position as :class:`BitarrayPosition`.

        A pair of bits represents a sentence:
        The first bit represents acceptance and the second bit rejection.
        Suspension of a sentence corresponds to both bits being False/0 and
        (minimal or flat) contradiction is present if both bits are True/1. For instance,
        the position :math:`\\{ s_1, s_3, \\neg s_4 \\}` is represented by :code:`10001001`.

        :returns: a bitarray representation of the position if possible, otherwise should return :code:`None`
        """
        pass

    @abstractmethod
    def as_ternary(self) -> int:
        """Position as ternary.

        The position :math:`\\mathcal{A}` represented by an integer of base 3 (at least).
        :math:`s_i \\in \\mathcal{A}` is represented by :math:`1*10^{i-1}`,
        :math:`\\neg s_i \\in \\mathcal{A}` by :math:`2*10^{i-1}` and
        :math:`s_i, \\neg s_i \\notin \\mathcal{A}` by zero. For instance, the position
        :math:`\\{ s_1, s_3, \\neg s_4 \\}` is represented by the integer 2101.

        .. note::
            Positions that are not minimally consistent cannot be represented in this way.

        :returns: a ternary representation of the position if possible, otherwise should return :code:`None`
        """
        pass

    @abstractmethod
    def as_set(self) -> Set[int]:
        """Position as integer set.

        The position :math:`\\mathcal{A}` represented by a python set of integer values.
        :math:`s_i \\in \\mathcal{A}` is represented by :math:`i` and :math:`\\neg s_i \\in \\mathcal{A}` by :math:`-i`.
        For instance, the position :math:`\\{ s_1, s_3, \\neg s_4 \\}` is represented by :code:`{1, 3, -4}`.

        :returns: a representation of the position as a set of integer values
        """
        pass

    @abstractmethod
    def as_list(self) -> List[int]:
        """Position as integer list.

        The position :math:`\\mathcal{A}` represented by a python list of integer values.
        :math:`s_i \\in \\mathcal{A}` is represented by :math:`i` and :math:`\\neg s_i \\in \\mathcal{A}` by :math:`-i`.
        For instance, the position :math:`\\{ s_1, s_3, \\neg s_4 \\}` is represented by :code:`[1, 3, -4]`.

        .. note::
            The returned order integer values is not specified.

        :returns: a representation of the position as a list of integer values
        """
        pass

    # ...

    # minimal consistency, analogue to minimal compatibility
    @abstractmethod
    def is_minimally_consistent(self) -> bool:
        """Checks for minimal consistency.

        A position :math:`\\mathcal{A}` is minimally consistent iff
        :math:`\\forall s \\in S: s\\in \\mathcal{A} \\rightarrow \\neg s \\notin \\mathcal{A}`

        :returns: :code:`True` iff the position is minimally consistent
        """
        pass

    # minimal compatibility of position with pos1, i.e. there is s such that s in position
    # and non-s in pos1, or vice-versa
    # ToDo: better 'is_minimally_compatible'?
    @abstractmethod
    def are_minimally_compatible(self, position: Position) -> bool:
        """Checks for minimal compatibility with :code:`position`.

        Two positions :math:`\\mathcal{A}` and :math:`\\mathcal{A}'` are minimally compatible iff
        :math:`\\mathcal{A} \\cup \\mathcal{A}'` is minimally consistent.

        :returns: :code:`True` iff the position-instance is compatible with :code:`position`
        """
        pass

    @abstractmethod
    def is_subposition(self, position: Position) -> bool:
        """Checks for set-theoretic inclusion.

        The position :math:`\\mathcal{A}` is a subposition of :math:`\\mathcal{A}'` iff
        :math:`\\mathcal{A} \\subset \\mathcal{A}'`.

        :returns: :code:`True` iff the position-instance is a subposition of :code:`position`
        """
        pass

    @abstractmethod
    def subpositions(self, n: int = -1, only_consistent_subpositions: bool = True) -> Iterator[Position]:
        """Iterator over subsets of size n.

        :param n: The size of the returned positions. If :code:`n` is :math:`-1`, the method returns all subpositions including the empty position and itself.
        :param only_consistent_subpositions: If :code:`True` only consistent subpositions will be returned.

        :return: A python iterator over subpositions of size n.
        """
        pass

    @abstractmethod
    def is_accepting(self, sentence: int) -> bool:
        """Checks whether :code:`sentence` is in the position.

        :param sentence: A sentence :math:`s_i` represented by i.

        :return: :code:`True` iff :code:`sentence` is in the position.
        """
        pass

    @abstractmethod
    def is_in_domain(self, sentence: int) -> bool:
        """Checks whether :code:`sentence` is in the domain of the position.

        :param sentence: A sentence :math:`s_i` represented by i.

        :return: :code:`True` iff :code:`sentence` is in the domain of the position.
        """
        pass

    @abstractmethod
    def size(self) -> int:
        """size of the position

        :return: The amount of sentences in that position (:math:`|S|`).
        """
        pass

    # ToDo: This is unfortunate. Since the method is static the use has to decide which implementation
    #  to use. Perhaps better as a non-static method?
    @staticmethod
    @abstractmethod
    def union(positions: Set[Position]) -> Position:
        """set-theoretic union

        :return: The set-theoretic union of the given set of :code:`positions`.
        """
        pass

    # ToDo: This is unfortunate. Since the method is static the user has to decide which implementation
    #  to use. Perhaps better as a non-static method?
    @staticmethod
    @abstractmethod
    def intersection(positions: Set[Position]) -> Position:
        """Intersect postions set-theoretically.

        :return: The set-theoretic intersection of the given set of :code:`positions`.
        """
        pass

    @abstractmethod
    def neighbours(self, depth: int) -> Iterator[Position]:
        """Neighbours of the position.

        Generates all neighbours of the position that can be reached by at most
        :code:`depth` many adjustments of individual sentences (including the position itself).
        The number of neighbours is
        :math:`\\sum_{k=0}^d {n\\choose k}*2^k)`, where n is the number of unnegated sentences and d is
        the depth of the neighbourhood.
        """


# ToDo: Discuss - getter for arguments? / perhaps removing of arguments?
class DialecticalStructure(ABC):
    """Class representing a dialectical structure.

        A dialectical structure is a tupel  :math:`\\left< S,A\\right>` of a sentence pool
        :math:`S = \\{ s_1, s_2, \\dots, s_N, \\neg s_1, \\neg s_2, \\dots, \\neg s_N \\}` and a set
        :math:`A` of arguments.

        An argument :math:`a=(P_a, c_a)` is defined as a pair consisting of premises :math:`P_a \\subset S` and
        a conclusion :math:`c_a \\in S`.
    """

    @staticmethod
    @abstractmethod
    def from_arguments(arguments: List[List[int]], n_unnegated_sentence_pool: int,
                       name : str = None) -> DialecticalStructure:
        """Instanciating a :class:`DialecticalStructure` from a list of int lists.

        :return: :class:`DialecticalStructure`
        """
        pass

    @abstractmethod
    def add_argument(self, argument: List[int]) -> DialecticalStructure:
        """Adds an argument to the dialectical structure.

        :param argument: An argument as an integer list. The last element represents the conclusion the others the premises.
        :return: The dialectical structure for convenience.
        """
        pass

    @abstractmethod
    def add_arguments(self, arguments: List[List[int]]) -> DialecticalStructure:
        """Adds arguments to the dialectical structure.

        :param arguments: A list of arguments as a list of integer lists.
        :return: The dialectical structure for convenience.
        """
        pass

    @abstractmethod
    def get_arguments(self) -> List[List[int]]:
        """The arguments as a list.

        :return: The arguments as a list of integer lists. The last element of each inner list represents the conclusion, the others the premises.
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get the name of the dialectical structure.

        :returns: The name of the dialectical structure as a string (default: \
        :code:`None`)
        """

        pass

    @abstractmethod
    def set_name(self, name: str):
        """Set the name of the dialectical structure."""

        pass

    '''
    Sentence-pool, domain and completeness
    '''

    @abstractmethod
    def is_complete(self, position: Position) -> bool:
        """Checks whether :code:`position' is complete.

        A position :math:`\\mathcal{A}` is complete iff the domain of :math:`\\mathcal{A}` is identical with the
        sentence pool :math:`S`.

        :return: :code:`True` iff the :code:`Position` is complete.
        """
        pass

    @abstractmethod
    def sentence_pool(self) -> Position:
        """Returns the sentences (without negations) of a dialetical structure's sentence pool.

        Returns from :math:`S = \\{ s_1, s_2, \\dots, s_N, \\neg s_1, \\neg s_2, \\dots, \\neg s_N \\}`
        only :math:`\\{ s_1, s_2, \\dots, s_N\\}`

        :return: "Half" of the full sentence pool :math:`S` as :code:`Position` (sentences without their negation).
        """
        pass

    '''
    Dialectic consistency & compatibility
    '''

    # dialectic consistency
    @abstractmethod
    def is_consistent(self, position: Position) -> bool:
        """Checks for dialectical consistency.

        A complete position :math:`\\mathcal{A}` is dialectically consistent iff it is a minimally consistent
        and for all arguments :math:`a=(P_a, c_a) \\in A` holds: If :math:`(\\forall p \\in P_a:p \\in \\mathcal{A})` then
        :math:`c_a \\in \\mathcal{A}`

        A partial position :math:`\\mathcal{A}` is dialectically consistent iff there is a complete and
        consistent position that extends :math:`\\mathcal{A}`.

        :return: :code:`True` iff :code:`position` it dialectically consistent.
        """
        pass

    @abstractmethod
    def are_compatible(self, position1: Position, position2: Position, ) -> bool:
        """Checks for dialectical compatibility of two positions.

        Two positions are dialectically compatible iff there is a complete and consistent positition that
        extends both.

        :return: :code:`True` iff :code:`position1` it dialectically compatible to :code:`position2`.
        """
        pass

    @abstractmethod
    def consistent_positions(self) -> Iterator[Position]:
        """ Iterator over all dialectically consistent positions.

        This iterator will include the empty position.

        :return: A python iterator over all dialectically consistent positions.
        """
        pass

    @abstractmethod
    def minimally_consistent_positions(self) -> Iterator[Position]:
        """ Iterator over all minimally consistent positions.

        A position :math:`\\mathcal{A}` is minimally consistent iff
        :math:`\\forall s \\in S: s\\in \\mathcal{A} \\rightarrow \\neg s \\notin \\mathcal{A}`


        This iterator will include the empty position.

        :return: An iterator over all minimally consistent positions.
        """

    @abstractmethod
    def consistent_complete_positions(self) -> Iterator[Position]:
        """ Iterator over all dialectically consistent and complete positions.

        :return: An iterator over all dialectically consistent and complete positions.
        """
        pass

    '''
    Dialectic entailment and dialectic closure of consistent positions
    '''

    @abstractmethod
    def entails(self, position1: Position, position2: Position) -> bool:
        """ Dialectical entailment.

        A position :math:`\\mathcal{A}` dialectically entails another position :math:`\\mathcal{B}` iff
        every consistent and complete position that extends :math:`\\mathcal{A}` also extends :math:`\\mathcal{B}`.


        :return: :code:`True` iff :code:`position2` is dialectically entailed by :code:`position1`.
        """
        pass

    @abstractmethod
    def closure(self, position: Position) -> Position:
        """Dialectical closure.

        The dialectical closure of a position :math:`\\mathcal{A}` is the intersection of all consistent and
        complete positions that extend :math:`\\mathcal{A}`. Note that in consequence, the empty position can have
        a non-empty closure.

        :return: The dialectical closure of :code:`position`.
        """
        pass

    @abstractmethod
    def is_closed(self, position: Position) -> bool:
        """Checks whether a position is dialectically closed.

        :return: :code:`True` iff :code:`position` is dialectically closed.
        """
        pass

    @abstractmethod
    def closed_positions(self) -> Iterator[Position]:
        """Iterator over all dialectically closed positions.

        This iterator will include the empty position, if it is closed.

        :return: A python-iterator over all dialectically closed positions.
        """
        pass

    # returns a list of minimal positions that entail *position*
    # Todo: fixed issue of there not being an axiomatic base in the source (now: returns None)
    # Todo: discuss - There is still some ambiguity in the description here. Right now we are chosing only
    # a position in source, if it is minimal. But should we perhaps confine the search for smaller positions to the
    # sources themselves?
    @abstractmethod
    def axioms(self, position: Position, source: Iterator[Position] = None) -> Iterator[Position]:
        """Iterator over all axiomatic bases from source.
        The source defaults to all consistent positions if it is not provided.

        A position :math:`\\mathcal{B}` is an axiomatic basis of another position :math:`\\mathcal{A}` iff
        :math:`\\mathcal{A}` is dialectically entailed by :math:`\\mathcal{B}` and there is no proper
        subset :math:`\\mathcal{C}` of :math:`\\mathcal{B}` such that :math:`\\mathcal{A}` is entailed by
        :math:`\\mathcal{C}`.

        This method should throw a :code:`ValueError` if the given position is inconsistent.

        :return: A python-iterator over all axiomatic bases of :code:`position` from :code:`source` and :code:`None` if
                there is no axiomatic basis in the source.
        """
        pass

    # there is no subset of position which entails position
    @abstractmethod
    def is_minimal(self, position: Position) -> bool:
        """Checks dialectical minimality.

        A position :math:`\\mathcal{A}` is dialectically minimal if every
        subposition :math:`\\mathcal{B}\\subseteq\\mathcal{A}` that
        entails :math:`\\mathcal{A}` is identical with :math:`\\mathcal{A}`.

        :return: :code:`True` iff :code:`position` is dialectically minimal.
        """
        pass

    @abstractmethod
    def minimal_positions(self) -> Iterator[Position]:
        """ Iterator over all dialectically minimal positions.

        :return: A python iterator over all dialectically minimal positions.
        """
        pass

    '''
    Counting extensions and DOJs 
    '''

    # sigma
    @abstractmethod
    # ToDo: ToDiscuss with Andi - If position in None the amount of all complete cons. positions should be returned.
    def n_complete_extensions(self, position: Position) -> int:
        """Number of complete and consistent extension.

        :return: The number of complete and consistent positions that extend :code:`position`.
        """
        pass

    @abstractmethod
    def degree_of_justification(self, position1: Position, position2: Position) -> float:
        """Conditional degree of justification.

        The conditional degree of justification :math:`DOJ` of two positions :math:`\\mathcal{A}` and
        :math:`\\mathcal{B}` is defined by :math:`DOJ(\\mathcal{A}| \\mathcal{B}):=\\frac{\\sigma_{\\mathcal{AB}}}{\\sigma_{\\mathcal{B}}}`
        with :math:`\\sigma_{\\mathcal{AB}}` the set of all consistent and complete positions that extend both :math:`\\mathcal{A}` and :math:`\\mathcal{B}`
        and :math:`\\sigma_{\\mathcal{B}}` the set of all consistent and complete positions that extend :math:`\\mathcal{B}`.

        :return: The conditional degree of justification of :code:`position1` with respect to :code:`position2`.
        """
        pass
