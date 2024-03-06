import pytest
from bitarray import bitarray
from random import randint
import importlib
import itertools
from typing import Set, List
import logging
import json
from os import getcwd, path, pardir, remove
import importlib
import tarfile


from .set_implementation import SetBasedPosition
from .base import Position
from .core import StandardPosition

# simply call `pytest -vv` on the console from directory of this file to execute the test
# or **`pytest -vv --log-cli-level INFO`** to show life logs (you simply use pytest logging mechanism, no need to
# configure a logger, see https://stackoverflow.com/questions/4673373/logging-within-pytest-tests)
# Use **`pytest tests.py -k 'position'`** to test only test cases in 'tests.py' (i.e. functions having
# the string 'test' in their name) that have (additionally) the string 'position' in their name.

model_implementations = [{'module_name': 'theodias',
                          'position_class_name':'StandardPosition',
                          'dialectical_structure_class_name': 'DAGDialecticalStructure'
                          },
                          # {'module_name': 'theodias',
                          #  'position_class_name':'StandardPosition',
                          #  'dialectical_structure_class_name': 'BDDDialecticalStructure',
                          #  },
                         {'module_name': 'theodias',
                          'position_class_name':'SetBasedPosition',
                          'dialectical_structure_class_name': 'DAGSetBasedDialecticalStructure'
                          },
                         {'module_name': 'theodias',
                          'position_class_name': 'NumpyPosition',
                          'dialectical_structure_class_name': 'DAGNumpyDialecticalStructure'
                          },
                          {'module_name': 'theodias',
                           'position_class_name': 'NumpyPosition',
                           'dialectical_structure_class_name': 'BDDNumpyDialecticalStructure',
                          },
                          {'module_name': 'theodias',
                           'position_class_name': 'BitarrayPosition',
                           'dialectical_structure_class_name': 'DAGBitarrayDialecticalStructure'
                          }
                         ]
# Helper functions


def get_dia(args: List[List[int]], n_unnegated_sentence_pool: int, impl):
    dia_class_ = getattr(importlib.import_module(impl['module_name']),
                              impl['dialectical_structure_class_name'])
    return dia_class_.from_arguments(args, n_unnegated_sentence_pool)

# this function will return a Position of the desired implementation
def get_position(pos: Set[int], n_unnegated_sentence_pool: int, impl):
    position_class_ = getattr(importlib.import_module(impl['module_name']),
                              impl['position_class_name'])
    return position_class_.from_set(pos, n_unnegated_sentence_pool)

# will return a list of the position in different implementation
def get_positions(pos: Set[int], n_unnegated_sentence_pool:int,
                  implementations = model_implementations) -> List[Position]:
    return [get_position(pos, n_unnegated_sentence_pool, impl) for impl in implementations ]

# returns a list of position tuples that is tuples of the  (pos1, pos2), with every combination for the
# different implementations
def get_implementations_product_of_positions(pos1: Set[int], pos2: Set[int],
                                             n_unnegated_sentence_pool:int,
                                             n_unnegated_sentence_pool_pos2: int = None,
                                             implementations = model_implementations):
    if n_unnegated_sentence_pool_pos2:
        return itertools.product(get_positions(pos1, n_unnegated_sentence_pool, implementations),
                                 get_positions(pos2, n_unnegated_sentence_pool_pos2, implementations))

    return itertools.product(get_positions(pos1, n_unnegated_sentence_pool, implementations),
                             get_positions(pos2, n_unnegated_sentence_pool, implementations))

def sentences_to_bitarray(sentences, n_unnegated_sentence_pool=None):
    if n_unnegated_sentence_pool:
        position_ba = 2 * n_unnegated_sentence_pool * bitarray('0')
    elif len(sentences) == 0:
        return bitarray()
    else:
        position_ba = 2 * max([abs(sentence) for sentence in sentences]) * bitarray('0')
    for sentence in sentences:
        if sentence < 0:
            position_ba[2 * (abs(sentence) - 1) + 1] = True
        else:
            position_ba[2 * (abs(sentence) - 1)] = True
    return position_ba

# The actual tests

class Testtheodias:

    log = logging.getLogger(__name__)

    # the following tests describe the expected behaviour for all implementations that
    # inherit `remodeldescription.basics.Position`
    def test_position_monadic_methods(self):
        for impl in model_implementations:
            self.log.info(f"Testing position implementations of types: {impl['position_class_name']}")

            ## TESTING MONADIC FUNCTIONS

            # testing function `sentence_pool`
            assert get_position({1, 2}, 3, impl).sentence_pool().size() == 3
            assert get_position({}, 3, impl).sentence_pool().size() == 3
            assert get_position({}, 0, impl).sentence_pool().size() == 0

            # testing function `domain`
            assert get_position({1, 2, 3}, 3, impl).domain().as_set() == {1, 2, 3, -1, -2, -3}
            assert get_position({1, 3}, 3, impl).domain().as_set() == {1, 3, -1, -3}
            assert get_position({1, -2, 3}, 3, impl).domain().as_set() == {1, 2, 3, -1, -2, -3}
            assert get_position({-2, -5}, 5, impl).domain().as_set() == {2, -2, 5, -5}

            # testing `as_bitarray
            assert get_position(set(),0, impl).as_bitarray() == bitarray('')
            assert get_position({-2, -5}, 5, impl).as_bitarray() == bitarray('0001000001')
            assert get_position({1, 2, 3}, 3, impl).as_bitarray() == bitarray('101010')
            assert get_position({1, 2, 3, -1}, 3, impl).as_bitarray() == bitarray('111010')
            assert get_position({1, 3 }, 3, impl).as_bitarray() == bitarray('100010')
            assert get_position({1, -2, 3}, 3, impl).as_bitarray() == bitarray('100110')

            # todo: check ternary for minimally inconsistent positions
            assert get_position({1, -3}, 3, impl).as_ternary() == 201
            assert get_position({1, -2, 3}, 3, impl).as_ternary() == 121
            assert get_position({1, 3 }, 3, impl).as_ternary() == 101
            assert get_position({1, 2, 3}, 3, impl).as_ternary() == 111
            # minimally inconsistent positions
            assert get_position({2, -2, -5}, 5, impl).as_ternary() == 20030
            #assert get_position({1, 2, 3, -1}, 3, impl).as_ternary() == None
            assert get_position({-2, -5}, 5, impl).as_ternary() == 20020
            assert get_position(set(), 0, impl).as_ternary() == 0

            assert get_position({-2, -5}, 5, impl).as_set() == {-2, -5}
            assert get_position({1, 2, 3}, 3, impl).as_set() == {1, 2, 3}
            assert get_position({1, 3 }, 3, impl).as_set() == {1, 3}
            assert get_position({1, -2, 3}, 3, impl).as_set() == {1, -2, 3}
            assert get_position(set(), 0, impl).as_set() == set()

            # testing `as_list`
            assert set(get_position({1, -2, 3}, 3, impl).as_list()) == {1, -2, 3}
            pos = get_position({1, 3 }, 3, impl)
            assert pos.as_list() == [3, 1] or pos.as_list() == [1, 3]
            pos = get_position({-2, -5}, 5, impl)
            assert pos.as_list() == [-2, -5] or pos.as_list() == [-5, -2]
            assert get_position(set(), 0, impl).as_list() == []

            assert get_position({1, 2, 3, -1}, 3, impl).is_minimally_consistent() == False
            assert get_position({1, 3}, 3, impl).is_minimally_consistent()
            assert get_position({1, -2, 3}, 3, impl).is_minimally_consistent()
            assert get_position({-2, -5}, 5, impl).is_minimally_consistent()
            assert get_position(set(), 0, impl).is_minimally_consistent()

            assert get_position({1, 2, 3}, 3, impl).is_accepting(2) == True
            assert get_position({1, 2, 3}, 3, impl).is_accepting(-1) == False
            assert get_position({1, 3 }, 3, impl).is_accepting(1) == True
            assert get_position({1, 3 }, 3, impl).is_accepting(-1) == False
            assert get_position({1, -2, 3}, 3, impl).is_accepting(3) == True
            assert get_position({1, -2, 3}, 3, impl).is_accepting(-1) == False
            assert get_position(set(), 0, impl).is_accepting(1) == False

            assert get_position({1, 2, 3}, 3, impl).is_in_domain(2) == True
            assert get_position({1, 2, 3}, 3, impl).is_in_domain(-1) == True
            assert get_position({1, 2, 3}, 3, impl).is_in_domain(-4) == False
            assert get_position({1, 3 }, 3, impl).is_in_domain(-1) == True
            assert get_position({1, 3 }, 3, impl).is_in_domain(2) == False
            assert get_position({1, -2, 3}, 3, impl).is_in_domain(-1) == True
            assert get_position({1, -2, 3}, 3, impl).is_in_domain(-4) == False
            assert get_position(set(), 0, impl).is_in_domain(-4) == False

            assert get_position({1, 2, 3}, 3, impl).size() == 3
            assert get_position({1, 3 }, 3, impl).size() == 2
            assert get_position({1, -2, 3}, 3, impl).size() == 3


            # testing subpositions
            assert ({frozenset(subpos.as_set()) for subpos in get_position({2, 3}, 3, impl).subpositions()} ==
                    {frozenset({2, 3}), frozenset({2}), frozenset({3}), frozenset()})
            assert ({frozenset(subpos.as_set()) for subpos in get_position({2, 3}, 3, impl).subpositions(n=0)} ==
                    {frozenset()})
            assert ({frozenset(subpos.as_set()) for subpos in get_position({2, -2}, 2, impl).subpositions()} ==
                    {frozenset({2}), frozenset({-2}), frozenset()})
            assert ({frozenset(subpos.as_set()) for subpos in get_position({2, -2}, 3, impl).subpositions(only_consistent_subpositions=False)} ==
                    {frozenset({2, -2}), frozenset({2}), frozenset({-2}), frozenset()})
            assert ({frozenset(subpos.as_set()) for subpos in get_position({2, -2}, 2, impl).subpositions(only_consistent_subpositions=False)} ==
                    {frozenset({2, -2}), frozenset({2}), frozenset({-2}), frozenset()})
            assert ({frozenset(subpos.as_set()) for subpos in get_position({2, -2, 3}, 3, impl).subpositions(n=2)} ==
                    {frozenset({2, 3}), frozenset({-2, 3})})
            assert ({frozenset(subpos.as_set()) for subpos in get_position({2, -2, 3}, 3, impl).subpositions(n=2, only_consistent_subpositions=False)} ==
                    {frozenset({2, -2}), frozenset({2, 3}), frozenset({-2, 3})})

            # testing `neighbours`
            # SetBasedPosition.from_set({}, 3),
            assert set(get_position({1, 2}, 3, impl).neighbours(1)) == {SetBasedPosition.from_set({1}, 3),
                                                                        SetBasedPosition.from_set({1,-2}, 3),
                                                                        SetBasedPosition.from_set({-1, 2}, 3),
                                                                        SetBasedPosition.from_set({2}, 3),
                                                                        SetBasedPosition.from_set({1, 2, 3}, 3),
                                                                        SetBasedPosition.from_set({1, 2, -3}, 3),
                                                                        SetBasedPosition.from_set({1, 2}, 3),}
            assert set(get_position({1}, 3, impl).neighbours(1)) == {SetBasedPosition.from_set({1}, 3),
                                                                     SetBasedPosition.from_set({1,-2}, 3),
                                                                     SetBasedPosition.from_set({1, 2}, 3),
                                                                     SetBasedPosition.from_set({1, 3}, 3),
                                                                     SetBasedPosition.from_set({1, -3}, 3),
                                                                     SetBasedPosition.from_set({-1}, 3),
                                                                     SetBasedPosition.from_set(set(), 3),}
            assert set(get_position(set(), 3, impl).neighbours(1)) == {SetBasedPosition.from_set(set(), 3),
                                                                       SetBasedPosition.from_set({1}, 3),
                                                                       SetBasedPosition.from_set({-1}, 3),
                                                                       SetBasedPosition.from_set({3}, 3),
                                                                       SetBasedPosition.from_set({-3}, 3),
                                                                       SetBasedPosition.from_set({2}, 3),
                                                                       SetBasedPosition.from_set({-2}, 3)}

    def test_position_set_operations(self):

        ################# TEST UNION #########################
        self.log.info(f"Testing set theoretic functions.")

        for impl in model_implementations:
            position_class_ = getattr(importlib.import_module(impl['module_name']),
                                      impl['position_class_name'])
            # union of empty sets
            assert position_class_.from_set(set(), 3).union(position_class_.from_set(set(), 3)).as_set() == set()
            assert (position_class_.from_set(set(), 3) | position_class_.from_set(set(), 3)).as_set() == set()
            # union of more than two positions
            assert position_class_.from_set({1,2}, 5).union(position_class_.from_set({1,2,3}, 5),
                                                            position_class_.from_set({1,2,3,4}, 5),
                                                            position_class_.from_set({3,4,5}, 5)
                                                            ) == position_class_.from_set({1,2,3,4,5}, 5)
            assert position_class_.from_set({1, 2}, 5) | position_class_.from_set({1, 2, 3}, 5) | \
                                                             position_class_.from_set({1, 2, 3, 4}, 5) | \
                                                             position_class_.from_set({3, 4, 5}, 5) \
                                                              == position_class_.from_set({1, 2, 3, 4, 5}, 5)
            # union with domain/(half) sentence pool
            pos = get_position({1, 2}, 3, impl)
            assert pos.union(pos.domain()) == get_position({1, 2, -1, -2}, 3, impl)
            assert pos | pos.domain() == get_position({1, 2, -1, -2}, 3, impl)

            assert pos.union(pos.sentence_pool()) == get_position({1, 2, 3}, 3, impl)
            assert pos.union(pos.sentence_pool()).domain() == get_position({1, 2, 3, -1, -2, -3}, 3, impl)

            # union with "none" should be handled as union with empty set
            assert pos.union() == pos
            # union should not change the sentence pool
            assert pos.union().sentence_pool() == pos.sentence_pool()
            assert pos.union(pos).sentence_pool() == pos.sentence_pool()

            # testing for different sentence-pool sizes: Should raise Error
            with pytest.raises(ValueError):
                pos.union(get_position({1, 2}, 4, impl))

        # testing union (including mixing implementations)
        for pos1, pos2 in get_implementations_product_of_positions({1, 3}, {1, 2, 3}, 3):
            assert pos1.union(pos2) == StandardPosition.from_set({1, 2, 3}, 3)
            # operator like
            assert pos1 | pos2 == StandardPosition.from_set({1, 2, 3}, 3)

            # union with nothing should be handled as union with empty set
            assert pos1.union() == pos1
            assert pos1 | StandardPosition.from_set(set(), 3) == pos1

        for pos1, pos2 in get_implementations_product_of_positions({-1, 2}, {2, 3}, 3):
            assert pos1.union(pos2).as_set() == {-1, 2, 3}
            assert (pos1 | pos2).as_set() == {-1, 2, 3}
            assert (pos1 | pos2) == StandardPosition.from_set({-1, 2, 3}, 3)

        for pos1, pos2 in get_implementations_product_of_positions({-1, 2}, set(), 3):
            assert pos1.union(pos2).as_set() == {-1, 2}
            assert (pos1 | pos2).as_set() == {-1, 2}
        # unions that result in minimally inconsistent positions (including mixing implementations)
        for pos1, pos2 in get_implementations_product_of_positions({-1, 2}, {1, 3}, 3):
            assert pos1.union(pos2).as_set() == {-1, 2, 3, 1}
            assert (pos1 | pos2).as_set() == {-1, 2, 3, 1}
            assert (pos1 | pos2) == StandardPosition.from_set({-1, 2, 3, 1}, 3)


        ################# TEST INTERSECTION #########################
        for impl in model_implementations:
            position_class_ = getattr(importlib.import_module(impl['module_name']),
                                      impl['position_class_name'])
            # intersection of empty sets
            assert position_class_.from_set(set(), 3).intersection(position_class_.from_set(set(), 3)).as_set() == set()
            assert (position_class_.from_set(set(), 3) & position_class_.from_set(set(), 3)).as_set() == set()
            # intersection of more than two positions
            assert position_class_.from_set({1, 2}, 5).intersection(position_class_.from_set({1, 2, 3}, 5),
                                                             position_class_.from_set({1, 2, 3, 4}, 5),
                                                             position_class_.from_set({2 ,3, 4, 5}, 5)
                                                             ) == position_class_.from_set({2}, 5)
            assert position_class_.from_set({1, 2}, 5) & position_class_.from_set({1, 2, 3}, 5) & \
                   position_class_.from_set({1, 2, 3, 4}, 5) & \
                   position_class_.from_set({2, 3, 4, 5}, 5) \
                   == position_class_.from_set({2}, 5)
            # intersection with domain/(half) sentence pool
            pos = get_position({1, 2}, 3, impl)
            assert pos.intersection(pos.domain()) == get_position({1, 2}, 3, impl)
            assert pos & pos.domain() == get_position({1, 2}, 3, impl)

            assert pos.intersection(pos.sentence_pool()) == get_position({1, 2}, 3, impl)
            assert pos.intersection(pos.sentence_pool()).domain() == get_position({1, 2, -1, -2}, 3, impl)

            assert pos.intersection(pos.sentence_pool().domain()) == get_position({1, 2,}, 3, impl)

            n = 6
            assert get_position({-3, 4, 5}, n, impl) & get_position({-3, -2}, n, impl) == get_position({-3}, n, impl)
            assert get_position({-3, 4, 5}, n, impl) & get_position({-3, -2}, n, impl).domain() == get_position({-3}, n, impl)
            assert get_position({-3, 4, 5}, n, impl) & get_position({3, -2}, n, impl).domain() == get_position({-3}, n, impl)

            # intersection with "none" should be handled as intersection with empty set
            assert pos.intersection().as_set() == set()
            # intersection should not change the sentence pool
            assert pos.intersection().sentence_pool() == pos.sentence_pool()
            assert pos.intersection(pos).sentence_pool() == pos.sentence_pool()

            # testing for different sentence-pool sizes: Should raise Error
            with pytest.raises(ValueError):
                pos.intersection(get_position({1, 2}, 4, impl))

        # testing intersection (including mixing implementations)
        for pos1, pos2 in get_implementations_product_of_positions({1, 3}, {1, 2, 3}, 3):
            assert pos1.intersection(pos2) == StandardPosition.from_set({1, 3}, 3)
            # operator like
            assert pos1 & pos2 == StandardPosition.from_set({1, 3}, 3)

            # intersection with nothing should be handled as intersection with empty set
            assert pos1.intersection() == StandardPosition.from_set(set(), 3)
            assert pos1 & StandardPosition.from_set(set(), 3) == StandardPosition.from_set(set(), 3)

        for pos1, pos2 in get_implementations_product_of_positions({-1, 2}, {2, 3}, 3):
            assert pos1.intersection(pos2).as_set() == {2}
            assert (pos1 & pos2).as_set() == {2}
            assert (pos1 & pos2) == StandardPosition.from_set({2}, 3)

        for pos1, pos2 in get_implementations_product_of_positions({-1, 2}, set(), 3):
            assert pos1.intersection(pos2).as_set() == set()
            assert (pos1 & pos2).as_set() == set()
        # intersections that result in minimally inconsistent positions (including mixing implementations)
        for pos1, pos2 in get_implementations_product_of_positions({-1, 1, 2}, {1, -1, 2, 3}, 3):
            assert pos1.intersection(pos2).as_set() == {-1, 2, 1}
            assert (pos1 & pos2).as_set() == {-1, 2, 1}
            assert (pos1 & pos2) == StandardPosition.from_set({-1, 2, 1}, 3)
        ################# TEST DIFFERENCE #########################
        for impl in model_implementations:
            position_class_ = getattr(importlib.import_module(impl['module_name']),
                                      impl['position_class_name'])
            # difference of empty sets
            assert position_class_.from_set(set(), 3).difference(position_class_.from_set(set(), 3)).as_set() == set()
            assert (position_class_.from_set(set(), 3) - position_class_.from_set(set(), 3)).as_set() == set()
            # difference of more than two positions
            assert position_class_.from_set({1, 2}, 4).difference(position_class_.from_set({2, 3}, 4)).difference(
                                                             position_class_.from_set({3, 4}, 4)) == position_class_.from_set({1}, 4)
            assert position_class_.from_set({1, 2}, 4) - position_class_.from_set({2, 3}, 4) - \
                   position_class_.from_set({3, 4}, 4) == position_class_.from_set({1}, 4)
            # difference with domain/(half) sentence pool
            pos = get_position({1, 2}, 3, impl)
            assert pos.difference(pos.domain()) == get_position(set(), 3, impl)
            assert pos - pos.domain() == get_position(set(), 3, impl)

            assert pos.difference(pos.sentence_pool()) == get_position(set(), 3, impl)
            assert pos.difference(pos.sentence_pool()).domain() == get_position(set(), 3, impl)

            assert pos.difference(pos.sentence_pool().domain()) == get_position(set(), 3, impl)

            # difference that results in minimally inconsistent positions
            assert position_class_.from_set({1, -1, 2}, 3) - position_class_.from_set({2,}, 3)  == position_class_.from_set({1, -1}, 3)

            n = 6
            assert get_position({-3, 4, 5}, n, impl) - get_position({-3, -2}, n, impl) == get_position({4,5}, n, impl)
            assert get_position({-3, 4, 5}, n, impl) - get_position({-3, -2}, n, impl).domain() == get_position({4, 5}, n, impl)
            assert get_position({-3, 4, 5}, n, impl) - get_position({3, -2}, n, impl).domain() == get_position({4, 5}, n, impl)

            # difference should not change the sentence pool
            assert pos.difference(pos).sentence_pool() == pos.sentence_pool()

            # testing for different sentence-pool sizes: Should raise Error
            with pytest.raises(ValueError):
                print(impl)
                pos.difference(get_position({1, 2}, 4, impl))

        # testing difference (including mixing implementations)
        for pos1, pos2 in get_implementations_product_of_positions({1, 3}, {1, 2, 3}, 3):
            assert pos1.difference(pos2) == StandardPosition.from_set(set(), 3)
            # operator like
            assert pos1 - pos2 == StandardPosition.from_set(set(), 3)

        for pos1, pos2 in get_implementations_product_of_positions({-1, 2}, {2, 3}, 3):
            assert pos1.difference(pos2).as_set() == {-1}
            assert (pos1 - pos2).as_set() == {-1}
            assert (pos1 - pos2) == StandardPosition.from_set({-1}, 3)

        for pos1, pos2 in get_implementations_product_of_positions({-1, 2}, set(), 3):
            assert pos1.difference(pos2).as_set() == {-1, 2}
            assert (pos1 - pos2).as_set() == {-1, 2}
        # differences that result in minimally inconsistent positions (including mixing implementations)
        for pos1, pos2 in get_implementations_product_of_positions({-1, 1, 2}, { 2, 3}, 3):
            assert pos1.difference(pos2).as_set() == {-1, 1}
            assert (pos1 - pos2).as_set() == {-1, 1}
            assert (pos1 - pos2) == StandardPosition.from_set({-1, 1}, 3)


    def test_position_static_methods(self):

        ## TESTING STATIC METHODS

        for impl in model_implementations:
            self.log.info(f"Testing static function of position implementation: {impl['position_class_name']}")
            position_class_ = getattr(importlib.import_module(impl['module_name']),
                                      impl['position_class_name'])
            # testing `from_set`
            assert (position_class_.from_set(set(), 0).as_set() == set())
            assert (position_class_.from_set(set(), 0).size() == 0)
            assert (position_class_.from_set(set(), 0).sentence_pool().size() == 0)
            assert (position_class_.from_set(set(), 2).size() == 0)
            assert (position_class_.from_set(set(), 2).sentence_pool().size() == 2)

            assert (position_class_.from_set({1, 2}, 2).as_set() == {1, 2})
            assert (position_class_.from_set({1, 2}, 2).size() == 2)
            assert (position_class_.from_set({1, 2}, 2).sentence_pool().size() == 2)

            assert (position_class_.from_set({1, 2}, 3).as_set() == {1, 2})
            assert (position_class_.from_set({1, 2}, 3).size() == 2)
            assert (position_class_.from_set({1, 2}, 3).sentence_pool().size() == 3)


    # the following test checks also whether the different implementions of Position are consistent to each other,
    # i.e., whether two-parameter functions can handle differing implementations
    def test_position_two_place_methods(self):
        self.log.info("Testing position implementations of two place relations:")

        # testing `==`
        for pos1, pos2 in get_implementations_product_of_positions({1, 3}, {1, 2, 3}, 3):
            assert pos1 != pos2
            assert pos2 != pos1
            assert pos1 != None
            assert None != pos1
            assert pos1 != 4
            assert 4 != pos1
            #assert pos1 != GlobalSetBasedReflectiveEquilibrium()
        for pos1, pos2 in get_implementations_product_of_positions({1, 3}, {3, 1}, 3):
            assert pos1 == pos2
            assert pos2 == pos1
        for pos1, pos2 in get_implementations_product_of_positions({1, 3}, {3, 1}, 5):
            assert pos1 == pos2
            assert pos2 == pos1
        # testing `hash` directly (it should be implementation-independent)
        for pos1, pos2 in get_implementations_product_of_positions({1, 3}, {3, 1}, 3):
            assert hash(pos1) == hash(pos2)
        for pos1, pos2 in get_implementations_product_of_positions({1, -1}, {-1, 1}, 3):
            assert hash(pos1) == hash(pos2)
        for pos1, pos2 in get_implementations_product_of_positions({1}, {-1, 1}, 3):
            assert hash(pos1) != hash(pos2)
        for pos1, pos2 in get_implementations_product_of_positions({-1}, {-1, 1}, 3):
            assert hash(pos1) != hash(pos2)

        for pos1, pos2 in get_implementations_product_of_positions(set(), set(), 3):
            assert hash(pos1) == hash(pos2)
        # positions with mismatching sentence pool should not be considered the same
        for pos1, pos2 in get_implementations_product_of_positions({1, 3}, {3, 1}, 3, 4):
            assert pos1 != pos2
            assert hash(pos1) != hash(pos2)
        # testing `hash` indirectly (comparison of sets and tuples draw on comparing hashkeys)
        for pos1, pos2 in get_implementations_product_of_positions({1, 3}, {3, 1}, 3):
            assert {pos1} == {pos2}
            assert (pos1,pos1) == (pos2,pos2)
        for pos1, pos2 in get_implementations_product_of_positions({1, 3}, {3, 1}, 3, 4):
            assert {pos1} != {pos2}
            assert (pos1,pos1) != (pos2,pos2)

        # all other two-place methods should raise a ValueError if the sentence pools do not match
        for pos1, pos2 in get_implementations_product_of_positions({1, 3}, {3, 1}, 3, 4):
            with pytest.raises(ValueError):
                pos1.is_minimally_compatible(pos2)


        # test `are_minimally_compatible`
        for pos1, pos2 in get_implementations_product_of_positions({1, 3}, {1, 2, 3}, 3):
            assert (pos1.is_minimally_compatible(pos2) == True)
            assert (pos2.is_minimally_compatible(pos1) == True)
        for pos1, pos2 in get_implementations_product_of_positions({1, -3}, {1, 2, 3}, 3):
            assert (pos1.is_minimally_compatible(pos2) == False)
            assert (pos2.is_minimally_compatible(pos1) == False)

        # test `is_subposition`
        for pos1, pos2 in get_implementations_product_of_positions({1, 3}, {1, 2, 3}, 3):
            assert pos1.is_subposition(pos2)
        for pos1, pos2 in get_implementations_product_of_positions({1, 2, 3}, {1, 2, 3}, 3):
            assert pos1.is_subposition(pos2)
        for pos1, pos2 in get_implementations_product_of_positions(set(), {1, 2, 3}, 3):
            assert pos1.is_subposition(pos2)
        for pos1, pos2 in get_implementations_product_of_positions({1, -2, 3}, {1, 2, 3}, 3):
            assert pos1.is_subposition(pos2) == False

    def test_position_setbased_implementation(self):
        assert (SetBasedPosition.from_set({1, 2}, 4).as_bitarray() == bitarray('10100000'))
        assert (SetBasedPosition({1, 2}, 4).as_bitarray() == bitarray('10100000'))
        assert (SetBasedPosition.from_set({}, 4).as_bitarray() == bitarray('00000000'))
        assert (SetBasedPosition({}, 4).as_bitarray() == bitarray('00000000'))
        assert (SetBasedPosition.from_set({-1, 2}, 4).as_bitarray() == bitarray('01100000'))
        assert (SetBasedPosition({-1, 2}, 4).as_bitarray() == bitarray('01100000'))

    def test_dialectical_structure_properties(self):
        for dia_impl in model_implementations:
            self.log.info(f"Testing dia structure of type: {dia_impl['dialectical_structure_class_name']}")
            # DIALECTICAL STRUCTURES
            # simple dia
            dia = get_dia([[1, 2], [3, -2]], 3, dia_impl)
            # simplest dia
            simple_dia = get_dia([[1, 2]], 2, dia_impl)
            # dia with no argument
            empty_dia = get_dia([], 3, dia_impl)
            # dia with no argument
            small_empty_dia = get_dia([], 2, dia_impl)

            # dialectical structure with tautologies
            taut_dia = get_dia([[1, 2], [-1, 2], [1, 3]], 3, dia_impl)
            # dia wit larger sentence pool than indicating by dialectical structure alone
            small_dia = get_dia([[1, 2], [2, 1]], 3, dia_impl)
            # dia with high inf. dens.
            dense_dia = get_dia([[1, 2], [2, 1], [3, -2], [-2, 3]], 3, dia_impl)

            # TESTING PROPERTIES OF THE DIA

            # Idea: without an argument the method returns number of all
            # maximally consistent positions
            assert (dia.n_complete_extensions() == 4)

            assert dia.sentence_pool().as_set() == {1, 2, 3}

            assert (len(list(dia.consistent_complete_positions())) == 4)
            assert ({pos for pos in empty_dia.consistent_complete_positions()} ==
                    {pos for pos in empty_dia.minimally_consistent_positions() if pos.size() == 3})
            small_dia.consistent_complete_positions()
            #assert ({frozenset(pos.as_set()) for pos in small_dia.consistent_complete_positions()} ==
            #        {frozenset({-3, -1, -2}), frozenset({3, -1, -2}), frozenset({1, 2, 3}), frozenset({1, 2, -3})})

            assert (len(list(dia.consistent_positions())) == 20)  # assuming that the empty position is counted
            assert (len(list(dia.closed_positions())) == 10)  # assuming that the empty position is counted

            assert ({pos for pos in empty_dia.consistent_positions()} ==
                    {pos for pos in empty_dia.minimally_consistent_positions()})

            # test `minimal_positions`
            assert set(simple_dia.minimally_consistent_positions()) == {
                SetBasedPosition({1}, 2),
                SetBasedPosition({-1}, 2),
                SetBasedPosition({2}, 2),
                SetBasedPosition({-2}, 2),
                SetBasedPosition({1, -2}, 2),
                SetBasedPosition({1, 2}, 2),
                SetBasedPosition({-1, 2}, 2),
                SetBasedPosition({-1, -2}, 2),
                SetBasedPosition(set(), 2)}
            # test `complete_minimal_positions`
            assert set(simple_dia.complete_minimally_consistent_positions()) == {
                SetBasedPosition({1, -2}, 2),
                SetBasedPosition({1, 2}, 2),
                SetBasedPosition({-1, 2}, 2),
                SetBasedPosition({-1, -2}, 2)}

            # `consistent_positions` (should include the empty position)
            assert set(small_empty_dia.consistent_positions()) == {SetBasedPosition({1}, 2),
                SetBasedPosition({-1}, 2),
                SetBasedPosition({2}, 2),
                SetBasedPosition({-2}, 2),
                SetBasedPosition({1, -2}, 2),
                SetBasedPosition({1, 2}, 2),
                SetBasedPosition({-1, 2}, 2),
                SetBasedPosition({-1, -2}, 2),
                SetBasedPosition(set(), 2)}
            assert set(small_empty_dia.consistent_positions(get_position({1}, 2, dia_impl))) == {
                                                                   SetBasedPosition({1}, 2),
                                                                   SetBasedPosition({1, -2}, 2),
                                                                   SetBasedPosition({1, 2}, 2),
                                                                   }
            assert set(small_empty_dia.consistent_positions(get_position(set(), 2, dia_impl))) == {
                                                                   SetBasedPosition({1}, 2),
                                                                   SetBasedPosition({-1}, 2),
                                                                   SetBasedPosition({2}, 2),
                                                                   SetBasedPosition({-2}, 2),
                                                                   SetBasedPosition({1, -2}, 2),
                                                                   SetBasedPosition({1, 2}, 2),
                                                                   SetBasedPosition({-1, 2}, 2),
                                                                   SetBasedPosition({-1, -2}, 2),
                                                                   SetBasedPosition(set(), 2)}
            assert set(dense_dia.consistent_positions()) == {
                                                             SetBasedPosition({1}, 3),
                                                             SetBasedPosition({-1}, 3),
                                                             SetBasedPosition({3, -2}, 3),
                                                             SetBasedPosition({3, -1, -2}, 3),
                                                             SetBasedPosition({1, 2}, 3),
                                                             SetBasedPosition({2}, 3),
                                                             SetBasedPosition({-2}, 3),
                                                             SetBasedPosition({-1, -2}, 3),
                                                             SetBasedPosition({-3}, 3),
                                                             SetBasedPosition({1, -3}, 3),
                                                             SetBasedPosition({2, -3}, 3),
                                                             SetBasedPosition({1, 2, -3}, 3),
                                                             SetBasedPosition({3}, 3),
                                                             SetBasedPosition({3, -1}, 3),
                                                             SetBasedPosition(set(), 3)}
            assert set(dense_dia.consistent_positions(get_position({1}, 3, dia_impl))) == {
                SetBasedPosition({1}, 3),
                SetBasedPosition({1, 2}, 3),
                SetBasedPosition({1, -3}, 3),
                SetBasedPosition({1, 2, -3}, 3),
                }
            assert set(dense_dia.consistent_positions(get_position(set(), 3, dia_impl))) == {
                SetBasedPosition({1}, 3),
                SetBasedPosition({-1}, 3),
                SetBasedPosition({3, -2}, 3),
                SetBasedPosition({3, -1, -2}, 3),
                SetBasedPosition({1, 2}, 3),
                SetBasedPosition({2}, 3),
                SetBasedPosition({-2}, 3),
                SetBasedPosition({-1, -2}, 3),
                SetBasedPosition({-3}, 3),
                SetBasedPosition({1, -3}, 3),
                SetBasedPosition({2, -3}, 3),
                SetBasedPosition({1, 2, -3}, 3),
                SetBasedPosition({3}, 3),
                SetBasedPosition({3, -1}, 3),
                SetBasedPosition(set(), 3)}

            # test `consistent_complete_positions`
            assert set(small_empty_dia.consistent_complete_positions()) == {
                                                                   SetBasedPosition({1, -2}, 2),
                                                                   SetBasedPosition({1, 2}, 2),
                                                                   SetBasedPosition({-1, 2}, 2),
                                                                   SetBasedPosition({-1, -2}, 2)}
            assert set(small_empty_dia.consistent_complete_positions(get_position(set(), 2, dia_impl))) == {
                SetBasedPosition({1, -2}, 2),
                SetBasedPosition({1, 2}, 2),
                SetBasedPosition({-1, 2}, 2),
                SetBasedPosition({-1, -2}, 2)}
            assert set(small_empty_dia.consistent_complete_positions(get_position({-2}, 2, dia_impl))) == {
                SetBasedPosition({1, -2}, 2),
                SetBasedPosition({-1, -2}, 2)}

            assert set(dense_dia.consistent_complete_positions()) == {
                                                             SetBasedPosition({3, -1, -2}, 3),
                                                             SetBasedPosition({1, 2, -3}, 3)}
            assert set(dense_dia.consistent_complete_positions(get_position(set(), 3, dia_impl))) == {
                SetBasedPosition({3, -1, -2}, 3),
                SetBasedPosition({1, 2, -3}, 3)}

            assert set(dense_dia.consistent_complete_positions(get_position({1, 2}, 3, dia_impl))) == {
                SetBasedPosition({1, 2, -3}, 3)}

            # test `closed_positions`
            assert set(small_empty_dia.closed_positions()) == {SetBasedPosition({1}, 2),
                                                                   SetBasedPosition({-1}, 2),
                                                                   SetBasedPosition({2}, 2),
                                                                   SetBasedPosition({-2}, 2),
                                                                   SetBasedPosition({1, -2}, 2),
                                                                   SetBasedPosition({1, 2}, 2),
                                                                   SetBasedPosition({-1, 2}, 2),
                                                                   SetBasedPosition({-1, -2}, 2),
                                                                   SetBasedPosition(set(), 2)}

            assert set(dense_dia.closed_positions()) == {    SetBasedPosition({3, -1, -2}, 3),
                                                             SetBasedPosition({1, 2, -3}, 3),
                                                             SetBasedPosition(set(), 3)}

    def test_dialectical_structure_sentence_pool_mismatch(self):
        # all methods should throw a ValueError if there is a mismatch between the sentence pools
        for impl in model_implementations:
            self.log.info(f"Testing dia structure of type: {impl['dialectical_structure_class_name']}")
            # simple dia
            dia = get_dia([[1, 2], [3, -2]], 3, impl)

            with pytest.raises(ValueError):
                dia.is_consistent(get_position({1, 2}, 2, impl))
            with pytest.raises(ValueError):
                dia.is_complete(get_position({1, 2}, 2, impl))
            with pytest.raises(ValueError):
                dia.are_compatible(get_position({1, 2}, 2, impl), get_position({1, 2}, 3, impl))
            with pytest.raises(ValueError):
                dia.entails(get_position({1, 2}, 2, impl), get_position({1, 2}, 3, impl))
            with pytest.raises(ValueError):
                dia.closure(get_position({1, 2}, 2, impl))
            with pytest.raises(ValueError):
                dia.is_closed(get_position({1, 2}, 2, impl))
            with pytest.raises(ValueError):
                dia.is_minimal(get_position({1, 2}, 2, impl))
            with pytest.raises(ValueError):
                dia.axioms(get_position({1, 2}, 2, impl))
            with pytest.raises(ValueError):
                dia.n_complete_extensions(get_position({1, 2}, 2, impl))
            with pytest.raises(ValueError):
                dia.degree_of_justification(get_position({1, 2}, 2, impl), get_position({1, 2}, 3, impl))
            with pytest.raises(ValueError):
                list(dia.consistent_positions(get_position({1, 2}, 2, impl)))
            with pytest.raises(ValueError):
                list(dia.consistent_complete_positions(get_position({1, 2}, 2, impl)))

    def test_dialectical_structure_monadic_methods(self):
        for dia_impl in model_implementations:
            self.log.info(f"Testing dia structure of type: {dia_impl['dialectical_structure_class_name']}")
            # DIALECTICAL STRUCTURES

            # simple dia
            dia = get_dia([[1, 2], [3, -2]], 3, dia_impl)
            # dia with not argument
            empty_dia = get_dia([], 3, dia_impl)
            # dialectical structure with tautologies
            taut_dia = get_dia([[1, 2], [-1, 2], [1,3]], 3, dia_impl)


            # TESTING MONADIC FUNCTIONS
            for impl in model_implementations:
                self.log.info(f"Testing dia structure with positions of type: {impl['position_class_name']}")

                # testing `is_complete`
                assert (dia.is_complete(get_position({1, 2}, 3, impl)) == False)
                assert (dia.is_complete(get_position({1, 2}, 3, impl)) == False)
                assert (dia.is_complete(get_position({1}, 3, impl)) == False)
                assert (dia.is_complete(get_position({-2}, 3, impl)) == False)
                assert (dia.is_complete(get_position({-1, -2, 3}, 3, impl)) == True)

                assert (empty_dia.is_complete(get_position({1, 2, 3}, 3, impl)))
                assert (empty_dia.is_complete(get_position({1, 2}, 3, impl)) == False)

                # testing `is_consisten`
                assert (dia.is_consistent(get_position({1, 2}, 3, impl)) == True)
                assert (dia.is_consistent(get_position({1, 2}, 3, impl)) == True)
                assert (dia.is_consistent(get_position({1}, 3, impl)) == True)
                assert (dia.is_consistent(get_position({-2}, 3, impl)) == True)
                assert (dia.is_consistent(get_position({-1, -2, 3}, 3, impl)) == True)
                assert (dia.is_consistent(get_position(set(), 3, impl)))

                assert (empty_dia.is_consistent(SetBasedPosition({2, 3}, 3)))

                #testing `closure`
                assert (dia.closure(get_position({1}, 3, impl)).as_set() == {1, 2, -3})
                assert (dia.closure(get_position({1, 2}, 3, impl)).as_set() == {1, 2, -3})
                assert (dia.closure(get_position({1, 2}, 3, impl)).as_set() == {1, 2, -3})
                assert (dia.closure(get_position({-2}, 3, impl)).as_set() == {-1, -2})
                assert (dia.closure(get_position({-1, -2, 3}, 3, impl)).as_set() == {-1, -2, 3})
                assert (dia.closure(get_position(set(), 3, impl)).size() == 0)
                # since {2} is a tautology the closure of the should be {2}
                assert (taut_dia.closure(get_position(set(), 3, impl)).as_set() == {2})
                # ex falso quodlibet
                assert (dia.closure(get_position({1, -1}, 3, impl)).as_set() == {1, 2, 3, -1 , -2, -3})
                assert (dia.closure(get_position({1, -2}, 3, impl)).as_set() == {1, 2, 3, -1 , -2, -3})

                # testing `is_closed`
                assert (dia.is_closed(get_position({1, 2}, 3, impl)) == False)
                assert (dia.is_closed(get_position({1, 2}, 3, impl)) == False)
                assert (dia.is_closed(get_position({1}, 3, impl)) == False)
                assert (dia.is_closed(get_position({-2}, 3, impl)) == False)
                assert (dia.is_closed(get_position({-1, -2, 3}, 3, impl)) == True)
                assert (dia.is_closed(get_position(set(), 3, impl)))
                # since {2} is a tautology the empty position shouldn't be closed
                assert (taut_dia.is_closed(get_position(set(), 3, impl)) == False)

                # testing `is_minimal`
                assert (dia.is_minimal(get_position({1, 2}, 3, impl)) == False)
                assert (dia.is_minimal(get_position({1, 2}, 3, impl)) == False)
                assert (dia.is_minimal(get_position({1}, 3, impl)) == True)
                assert (dia.is_minimal(get_position({-2}, 3, impl)) == True)
                assert (dia.is_minimal(get_position({-1, -2, 3}, 3, impl)) == False)
                # the only subset of the empty position is the empty position:
                assert (dia.is_minimal(get_position(set(), 3, impl)))
                assert (dia.is_minimal(get_position({-1}, 3, impl)))
                assert (dia.is_minimal(get_position({3}, 3, impl)))
                assert (dia.is_minimal(get_position({3, -2}, 3, impl)) == False)
                assert (dia.is_minimal(get_position({-1, -3}, 3, impl)))
                assert (dia.is_minimal(get_position({-1, 1}, 3, impl)))


                # testing `n_complete_extension`
                assert (dia.n_complete_extensions(get_position({1, 2}, 3, impl)) == 1)
                assert (dia.n_complete_extensions(get_position({1, 2}, 3, impl)) == 1)
                assert (dia.n_complete_extensions(get_position({1}, 3, impl)) == 1)
                assert ({frozenset(axiom.as_set()) for axiom in dia.axioms(get_position({-2}, 3, impl))} == {frozenset({-2}), frozenset({3})})
                assert (dia.n_complete_extensions(get_position({-2}, 3, impl)) == 2)
                assert (dia.n_complete_extensions(get_position({-1, -2, 3}, 3, impl)) == 1)
                assert (dia.n_complete_extensions(get_position(set(), 3, impl)) ==
                        len([pos for pos in dia.consistent_complete_positions()]))
                # if no position is given, it should return the number of all complete consistent positions
                assert (dia.n_complete_extensions() ==
                        len([pos for pos in dia.consistent_complete_positions()]))
                # if the given position is inconsistent, there shouldn't be any consistent extensions
                assert (dia.n_complete_extensions(get_position({-1, 1}, 3, impl)) == 0)
                assert (dia.n_complete_extensions(get_position({1, -2}, 3, impl)) == 0)


                # if no position is given, it should return the number of all complete consistent positions
                assert (empty_dia.n_complete_extensions() == 8)
                assert (empty_dia.n_complete_extensions(get_position({1, 2}, 3, impl)) == 2)
                assert (empty_dia.n_complete_extensions(get_position({-1}, 3, impl)) == 4)
                assert (empty_dia.n_complete_extensions(get_position({-1, 1}, 3, impl)) == 0)
                assert (empty_dia.n_complete_extensions(get_position(set(), 3, impl)) == 8)

                # testing `axioms`
                assert set(dia.axioms(get_position({1, 2}, 3, impl))) ==  {SetBasedPosition({1}, 3)}
                assert set(dia.axioms(get_position({1}, 3, impl))) == {SetBasedPosition({1}, 3)}
                assert set(dia.axioms(get_position({-2}, 3, impl))) ==  {SetBasedPosition({-2}, 3),
                                                                         SetBasedPosition({3}, 3)}
                assert set(dia.axioms(get_position({-2}, 3, impl),
                                      [get_position({3}, 3, impl)])) ==  {SetBasedPosition({3}, 3)}

                assert set(dia.axioms(get_position({-1, -2, 3}, 3, impl))) == {SetBasedPosition({3}, 3)}
                assert set(dia.axioms(get_position(set(), 3, impl))) == {SetBasedPosition(set(), 3)}

                assert set(dia.axioms(get_position({-1}, 3, impl))) == {SetBasedPosition({-1}, 3),
                                                                        SetBasedPosition({3}, 3),
                                                                        SetBasedPosition({-2}, 3)}
                assert set(dia.axioms(get_position({-1}, 3, impl),
                                      [get_position({3}, 3, impl)])) == {SetBasedPosition({3}, 3)}
                assert set(dia.axioms(get_position({-1}, 3, impl),
                                      [get_position({-1}, 3, impl)])) == {SetBasedPosition({-1}, 3)}
                assert set(dia.axioms(get_position({-1}, 3, impl),
                                      [get_position({-1}, 3, impl),
                                       get_position({2}, 3, impl)])) == {SetBasedPosition({-1}, 3)}

                assert set(dia.axioms(get_position({-1}, 3, impl),
                                      [get_position({3}, 3, impl),
                                       get_position({-2}, 3, impl)])) == {SetBasedPosition({3}, 3),
                                                                         SetBasedPosition({-2}, 3)}
                # we (indirectly) check whether the return has duplicate values (which it shouldn't)
                assert len(list(dia.axioms(get_position({-1}, 3, impl),
                                      [get_position({3}, 3, impl),
                                       get_position({-2}, 3, impl),
                                       get_position({3}, 3, impl),
                                       get_position({-2}, 3, impl)]))) == 2
                # returns [] if there is no axiomatic base in `sources`
                # Here, {3,2} entails {-1} but is not (globally) minimal
                assert dia.axioms(get_position({-1}, 3, impl),
                                      [get_position({3,2}, 3, impl)]) == []
                # Here, {-1,2} entails {-1} but is not (globally) minimal
                assert dia.axioms(get_position({-1}, 3, impl),
                                      [get_position({-1, 2}, 3, impl)]) == []
                assert dia.axioms(get_position({-1}, 3, impl),
                                  [get_position({2}, 3, impl)]) == []
                assert dia.axioms(get_position({-1}, 3, impl),
                                  [get_position(set(), 3, impl)]) == []

                # Inconsistent position that has a smaller base (which is not in source)
                assert dia.axioms(get_position({-1}, 3, impl),
                                      [get_position({-2, 2}, 3, impl)]) == []
                # Inconsistent position in source which is minimal (ex falso quodlibet)
                assert set(dia.axioms(get_position({1}, 3, impl),
                                      [get_position({-2,2}, 3, impl)])) == {SetBasedPosition({-2, 2}, 3)}

                assert set(dia.axioms(get_position({1, 2, -3}, 3, impl))) == {SetBasedPosition({1}, 3)}

                # since {2} is a tautology is can be axiomatized by the empty set
                assert set(taut_dia.axioms(get_position({2}, 3, impl))) == {SetBasedPosition(set(), 3)}

                # An inconsistent Position cannot be axiomatized
                with pytest.raises(ValueError):
                    dia.axioms(get_position({-1, 1}, 3, impl))
                with pytest.raises(ValueError):
                    dia.axioms(get_position({1, -2}, 3, impl))




    def test_dialectical_structure_two_place_methods(self):

        for dia_impl in model_implementations:
            self.log.info(f"Testing dia structure of type: {dia_impl['dialectical_structure_class_name']}")

            # DIALECTICAL STRUCTURES
            # simple dia
            dia = get_dia([[1, 2], [3, -2]], 3, dia_impl)
            # dia without args
            empty_dia = get_dia([], 3, dia_impl)
            # dialectical structure with tautologies
            taut_dia = get_dia([[1, 2], [-1, 2], [1,3]], 3, dia_impl)

            # testing `degree_of_justification`
            for pos1, pos2 in get_implementations_product_of_positions({2}, {-3}, 3):
                assert (dia.degree_of_justification(pos2, pos1) == 1.0)
                assert (dia.degree_of_justification(pos1, pos2) == 2 / 3)
            for pos1, pos2 in get_implementations_product_of_positions({-1}, {-1, 1}, 3):
                with pytest.raises(ZeroDivisionError):
                    dia.degree_of_justification(pos1, pos2)

            # testing `are_compatible`
            for pos1, pos2 in get_implementations_product_of_positions({2}, {-3}, 3):
                assert (dia.are_compatible(pos1, pos2) == True)
            for pos1, pos2 in get_implementations_product_of_positions(set(), {-1}, 3):
                assert (dia.are_compatible(pos1, pos2))
                assert (dia.are_compatible(pos2, pos1))
            for pos1, pos2 in get_implementations_product_of_positions(set(), {-1, 1}, 3):
                assert (dia.are_compatible(pos2, pos1) == False)
                assert (dia.are_compatible(pos1, pos2) == False)
            for pos1, pos2 in get_implementations_product_of_positions(set(), set(), 3):
                assert (dia.are_compatible(pos1, pos2))
            for pos1, pos2 in get_implementations_product_of_positions({1}, {-1}, 3):
                assert (dia.are_compatible(pos1, pos2) == False)

            # testing `are_compatible` w.r.t. an empty dialectical structure
            for pos1, pos2 in get_implementations_product_of_positions({-1},{2} , 3):
                assert (empty_dia.are_compatible(pos1, pos2))
            for pos1, pos2 in get_implementations_product_of_positions({-1, 2}, {2}, 3):
                assert (empty_dia.are_compatible(pos1, pos2))
            for pos1, pos2 in get_implementations_product_of_positions({-1, 2}, {2, 3}, 3):
                assert (empty_dia.are_compatible(pos1, pos2))
            for pos1, pos2 in get_implementations_product_of_positions({-1, 2},{-2, 3} , 3):
                assert (empty_dia.are_compatible(pos1, pos2)==False)

            # testing `entails`
            for pos1, pos2 in get_implementations_product_of_positions({2}, {-3}, 3):
                assert (dia.entails(pos1, pos2) == True)
                assert (dia.entails(pos2, pos1) == False)
            for pos1, pos2 in get_implementations_product_of_positions(set(), set(), 3):
                assert (dia.entails(pos1, pos2))
            for pos1, pos2 in get_implementations_product_of_positions(set(), {1}, 3):
                assert (dia.entails(pos1, pos2) == False)
            for pos1, pos2 in get_implementations_product_of_positions({-2}, {-1, -3}, 3):
                assert (dia.entails(pos1, pos2) == False)
            for pos1, pos2 in get_implementations_product_of_positions({-2}, {-1}, 3):
                assert (dia.entails(pos1, pos2))
            for pos1, pos2 in get_implementations_product_of_positions({2}, {-3}, 3):
                assert (dia.entails(pos1, pos2))
            # but also: conjunctions entails their conjuncts:
            for pos1, pos2 in get_implementations_product_of_positions({3, -1, -2}, {3}, 3):
                assert (dia.entails(pos1, pos2))
            for pos1, pos2 in get_implementations_product_of_positions({-1}, set(), 3):
                # empty position is entailed by any other position
                assert (dia.entails(pos1, pos2))
            for pos1, pos2 in get_implementations_product_of_positions({-1}, {-2, 2}, 3):
                # consistent positions do not entail inconsistencies
                assert (dia.entails(pos1, pos2) == False)
            for pos1, pos2 in get_implementations_product_of_positions({-1, 1}, {-1, 2, 3}, 3):
                # ex falso quodlibet
                assert (dia.entails(pos1, pos2))
            for pos1, pos2 in get_implementations_product_of_positions({2, 3}, {1}, 3):
                # ex falso quodlibet
                assert (dia.entails(pos1, pos2) == True)
            for pos1, pos2 in get_implementations_product_of_positions( {1}, {2, 3}, 3):
                #consistent positions do not entail inconsistent positions
                assert (dia.entails(pos1, pos2) == False)
            for pos1, pos2 in get_implementations_product_of_positions( {2}, {-1,1}, 3):
                #consistent positions do not entail inconsistent positions
                assert (dia.entails(pos1, pos2) == False)

            #  test `entails` in empty dialectical structures
            for pos1, pos2 in get_implementations_product_of_positions({-1}, {2}, 3):
                assert (empty_dia.entails(pos1, pos2) == False)
            for pos1, pos2 in get_implementations_product_of_positions({-1, 2}, {2}, 3):
                assert (empty_dia.entails(pos1, pos2))
            for pos1, pos2 in get_implementations_product_of_positions({1, 2, 3},{2, 3} , 3):
                assert (empty_dia.entails(pos1, pos2))
            for pos1, pos2 in get_implementations_product_of_positions({2, 3}, {1, 2, 3}, 3):
                assert (empty_dia.entails(pos1, pos2) == False)
            # since {2} is dia-tautological it shouldbe entaild be the empty set
            for pos1, pos2 in get_implementations_product_of_positions(set(), {2}, 3):
                assert (taut_dia.entails(pos1, pos2))
