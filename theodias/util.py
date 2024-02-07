"""
A collection of convenient helper-functions.

"""
from __future__ import annotations

from .base import DialecticalStructure, Position

import os
import pickle
from pysat.solvers import Glucose3, Minisat22
from pysat.formula import CNF
import math
from typing import List, Iterator, Set
from random import randint, choices, choice, sample
import importlib
from json import JSONEncoder, dumps, dump, loads, load
import numpy as np
from deprecated import deprecated

# todo:
def is_satisfiable(args: List[List[int]], principles: List[int] = None):
    """
    Fast method of checking for satisfiability (using xxx).
    """
    cnf = CNF()
    # convert arguments to disjunctive clauses
    for clause in [arg_to_cnf(arg) for arg in args]:
        cnf.append(clause)

    # add sentences from principles as additional clauses
    if principles:
        for principle in principles:
            cnf.append([principle])

    return Glucose3(cnf).solve()


def inferential_density(dialectical_structure):
    """ Inferential density of a dialectical structure :math:`\tau`.

    As defined by Betz (2013, Debate Dynamics, p. 44): :math:`D(\\tau) = \\frac{n-log_{2}(\\sigma)}{n}`,
    where :math:`\\sigma(\\tau)` is the number of complete and dialectically consistent positions
    of :math:`\tau`.
    """
    n_complete_extensions = dialectical_structure.n_complete_extensions()
    if n_complete_extensions == 0:
        raise ValueError("The given dialectical structure is not satisfiable: There are no complete and dialectically consistent positions on this structure.")
    return (dialectical_structure.sentence_pool().size() -
            math.log2(n_complete_extensions)) / dialectical_structure.sentence_pool().size()


# todo: unit test
def get_principles(arguments):
    """"
    Returns a list of tuples of the form :code:`(principle, multiplicity)` with :code:`multiplicity`
    indicating the multiplicity of the principle.
    A sentence counts as a principle iff it occurs in at least one argument as premise and it or its negation
    does not occur as a conclusion in an argument. The multiplicity counts in how many argument the sentence occurs
    as premise.
    """
    sentences = set([item for subl in arguments for item in subl])
    principles = [(sentence, sum([(sentence in arg[:-1]) for arg in arguments])) for sentence in sentences if
                  sum([(sentence in arg[:-1]) for arg in arguments]) > 0 and
                  # the sentence or its negation is not a conclusion
                  sum([(sentence == arg[-1]) or (-sentence == arg[-1]) for arg in arguments]) == 0]
    return principles


def number_of_complete_consistent_positions(args: List[List[int]], n_unnegated_sentence_pool: int) -> int:
    """
    Fast method to calculate the number of complete consistent positions.
    """
    cnf = args2cnf(args, n_unnegated_sentence_pool)
    return len(list(Minisat22(cnf.clauses).enum_models()))


def arg_to_cnf(arg: List[int]):
    """Converts list representation of an argument to a cnf-position-like list
    of negated premises + non-negated conclusion"""
    return [-prem for prem in arg[:-1]] + [arg[-1]]


def args2cnf(arguments:List[List[int]], n_sentence_pool:int) -> CNF:
    """Converting argument list to :code:`pysat.formula.CNF`

    This method converts a list of arguments into a conjunctive normal form.
    Sentences :math:`p` that are not used in arguments will be added in a tautological manner (i.e.,
    :math:`p\\vee\\neg p`)
    to the CNF formula.

    """
    cnf = CNF()
    for clause in [arg_to_cnf(argument) for argument in arguments]:
        cnf.append(clause)
    # flat list of all sentences appearing in arguments
    cnf_sentences = {item for sublist in cnf.clauses for item in sublist}
    sentence_pool = [i for i in range(-n_sentence_pool, n_sentence_pool + 1) if not i == 0]
    # sentences from the sentence pool if neither they nor their negation appear in arguments
    missing_sentences = [sentence for sentence in sentence_pool if
                         sentence not in cnf_sentences and -sentence not in cnf_sentences]
    # adding those as disjunction to the cnf
    if missing_sentences:
        cnf.append(missing_sentences)
    return cnf

def write_as_tex(directory, fileName, arg_list):
    """
    ToDo: doca
    """
    with open(os.path.join(directory, fileName), "w") as fi:
        fi.write("\\documentclass[tikz,border=10pt]{standalone}\n")
        fi.write("\\begin{document}\n")
        fi.write(
            "\\begin{tikzpicture}[node distance=2cm, >=stealth, arg/.style={rectangle, draw}, attack/.style={->, dashed, bend right}, support/.style={->, bend right}]\n")
        # ToDo: Add repeated conclusions (pos or neg) as stand-alone theses and their relations to arguments

        for i in range(len(arg_list)):
            text = "\\draw ({}*360/{}: {}*0.5cm) node[arg]({}){{\\begin{{tabular}}{{c}}".format(i + 1, len(arg_list),
                                                                                                len(arg_list), i)
            for prem in arg_list[i][:-1]:
                text += '{} \\\\'.format(prem)
            text += "\\hline {}".format(arg_list[i][-1])

            text += "\\end{tabular}};\n"
            fi.write(text)

        text = ''
        for i in range(len(arg_list)):
            con = arg_list[i][-1]
            for j in range(len(arg_list)):

                if con in arg_list[j][:-1]:
                    text += "\\draw [support] ({}) edge ({});\n".format(i, j)
                elif -con in arg_list[j][:-1]:
                    text += "\\draw [attack] ({}) edge ({});\n".format(i, j)

        fi.write(text)

        fi.write("\\end{tikzpicture}\n")
        fi.write("\\end{document}")



def write_as_dot(directory, fileName, arg_list, equal_rank_for_principles = False):
    """
    ToDo: doc
    """
    sentences = set([item for subl in arg_list for item in subl])
    # prop that occur more than once as premises or more than once as conclusions
    theses = [sentence for sentence in sentences if
              sum([(sentence in arg[:-1]) or (-sentence in arg[:-1]) for arg in arg_list]) > 1 or
              sum([(sentence == arg[-1]) or (-sentence == arg[-1]) for arg in arg_list]) > 1]
    # filter theses whose positive negations are already in the set
    theses = [sentence for sentence in theses if not ((-sentence in theses) and sentence < 0)]

    if equal_rank_for_principles:
        principles = get_principles(arg_list)
        same_rank_line = "{rank = same;"

    with open(os.path.join(directory, fileName), "w") as fi:
        fi.write("digraph G{\n")
        fi.write("rankdir=TB;\n")
        fi.write("node [shape=rectangle, style=\"argument\", width=0, height=0, margin=0.05];\n")
        fi.write("edge [lblstyle=\"xshift=-6\"];\n")
        # ToDo: Add repeated conclusions (pos or neg) as stand-alone theses and their relations to arguments

        # argument-nodes
        for i in range(len(arg_list)):
            text = "NA{} [texlbl=\"\\begin{{tabular}}{{c}}".format(i)
            for prem in arg_list[i][:-1]:
                text += '{} \\\\'.format(prem)
            text += "\\hline {}".format(arg_list[i][-1])
            text += "\\end{tabular}\"];\n"
            fi.write(text)


        # theses-nodes
        for i in range(len(theses)):
            text = "NT{} [texlbl=\"{}\"];\n".format(i, theses[i])
            fi.write(text)
            if equal_rank_for_principles and (theses[i] in principles or -theses[i] in principles):
                same_rank_line += "NT{};".format(i)

        if equal_rank_for_principles:
            fi.write(same_rank_line + "}\n")


        text = ''
        for i in range(len(arg_list)):
            con = arg_list[i][-1]
            # relations from args to args
            for j in range(len(arg_list)):
                if con in arg_list[j][:-1] and (con not in theses) and (-con not in theses):
                    text += "NA{} -> NA{} [style =\"support\"];\n".format(i, j)
                elif -con in arg_list[j][:-1] and (-con not in theses) and (con not in theses):
                    text += "NA{} -> NA{} [style =\"attack\"];\n".format(i, j)
            # relations from args to theses
            for j in range(len(theses)):
                if con == theses[j]:
                    text += "NA{} -> NT{} [style =\"support\"];\n".format(i, j)
                if -con == theses[j]:
                    text += "NA{} -> NT{} [style =\"attack\"];\n".format(i, j)
        # relations from thesis to args
        for i in range(len(theses)):
            thesis = theses[i]
            for j in range(len(arg_list)):
                if thesis in arg_list[j][:-1]:
                    text += "NT{} -> NA{} [style =\"support\"];\n".format(i, j)
                if -thesis in arg_list[j][:-1]:
                    text += "NT{} -> NA{} [style =\"attack\"];\n".format(i, j)


        fi.write(text)
        fi.write("}")


# converts a 10-based number into it ternary representation
def __dec2ternary(pos, n_sentences):
    if pos == 0:
        return [0 for i in range(n_sentences)]
    digits = []
    while pos:
        digits.append(int(pos % 3))
        pos //= 3
    return [0 for i in range(n_sentences - len(digits))] + digits[::-1]


def __ternary2set(pos_ternary_list, n_sentences) -> Set[int]:
    return {i if pos_ternary_list[-i] == 1 else -i for i in range(1, n_sentences + 1) if pos_ternary_list[-i] != 0}


def save_dialectical_structure(DIR, fileName, dialectical_structure):
    """
     ToDo: doc
     """

    with open(os.path.join(DIR, fileName), 'wb') as f:
        pickle.dump(dialectical_structure, f, pickle.HIGHEST_PROTOCOL)

    return None


def load_dialectical_structure(DIR, fileName):
    """
     ToDo: doc
     """

    with open(os.path.join(DIR, fileName), 'rb') as f:
        dialectical_structure = pickle.load(f)

    return dialectical_structure


# todo: parameter for fixed sentence size
def random_positions(n_sentences: int, k: int = 1, allow_empty_position: bool = False) -> List[Set[int]]:
    """Generation of random positions.

    Generates random minimally consistent positions in their set-representation.

    :param int n_sentences: Size of the sentence-pool (without negations).
    :param int k: Sample size.
    :param bool allow_empty_position: Iff :code:`True` an empty position might be returned.
    :return: A list of random positions is integer sets.
    :rtype: List[Set[int]]
    :raises ValueError: if k exceeds the number of minimally consistent positions given
        :code:`n_sentences`.
    """
    # number of minimally consistent positions
    n_minimally_consistent_pos = 3 ** n_sentences
    # choose random position (by its int/decimal-representation)
    if allow_empty_position:
        positions = sample(range(n_minimally_consistent_pos), k = k)
    else:
        positions = sample(range(1, n_minimally_consistent_pos), k = k)
    # convert the pos to its ternary representation and then to its set representation
    return [__ternary2set(__dec2ternary(pos, n_sentences), n_sentences) for
            pos in positions]

@deprecated(reason="This method is deprecated. Use 'random_positions()' instead.")
def random_position_as_set(n_sentences, allow_empty_position=False) -> Set[int]:
    """ Generates a random minimally consistent position in its set-representation.

    Args:
        n_sentences (int): Size of the sentence-pool (without negations).
        allow_empty_position (bool): Iff :code:`True` an empty position might be returned.

    Returns:
        A set of integer representing the position.
    """
    # number of minimally consistent positions
    n_minimally_consistent_pos = 3 ** n_sentences
    # choose random position (by its int/decimal-representation)
    if allow_empty_position:
        pos = randint(0, n_minimally_consistent_pos - 1)
    else:
        pos = randint(1, n_minimally_consistent_pos - 1)
    # convert the pos to its ternary representation and then to its set representation
    return __ternary2set(__dec2ternary(pos, n_sentences), n_sentences)

def create_random_arguments(n_sentences, n_arguments, n_max_premises, n_principles=0,
                            variation=True,
                            n_premises_weights=None,
                            connected=True, use_all_sentences = False,
                            max_loops = 1000) -> List[List[int]]:
    """Returns a list of arguments represented as integer lists.

    The dialectical structure given by the returned arguments is satisfiable
    (i.e. there is at least one complete and dialectically consistent position).
    Furthermore, the arguments avoid begging the question, repeating premises,
    flat contradictions among premises and with the conclusion, and using the same
    premises for different conclusions.

    Args:
        n_sentences (int): Number of non-negated sentences of the sentence pool.
            Every sentence (or its negation) occurs at least once in an argument.
        n_arguments (int): Number of generated arguments.
        n_max_premises (int) : Maximal number of premises per argument.
        n_principles (int): The number of principles, i.e. sentences that may occur in arguments as premises only.
        variation (bool): If true (default), the number of premises per argument is chosen randomly.
            Otherwise it is constantly n_max_premises.
        n_premises_weights (list of floats): If :code:`None` (default) the distribution of randomly choosing the number of
            premises in an argument is uniform. Otherwise the given weights will be used to choose a number of premises
            between 1 and :code:`n_max_premises`.
        connected (bool): If true (default), the arguments are related
            (either by attack or support) to at least one other argument in the structure.
        use_all_sentences (bool): If true, the algorithm returns only dialectical structures that include each sentence
            ,or its negation respectively, in the arguments. If the algorithm is not able to generate such a structure
            in the limits given by :code:`max_loops`, a RuntimeWarning is thrown.
        max_loops (int): Breaks possibly endless loops. Number of arguments that tested
            whether they fit into the dialectical structure. If the algorithm takes longer a RuntimeWarning is thrown.

    Returns:
        A list of argument as integer lists.

    """

    # monitor unused (unnegated) sentences in order to ensure that every sentence (or its negation)
    # occurs at least once if there are sufficiently many arguments and/or premises per argument

    unused_sentences = set(range(1, n_sentences + 1))
    # principles are sentences that are to appear only as premises in arguments, never as conclusions:

    principles = set(s for s in range(1, n_principles+1))

    def weighted_sample_without_replacement(population, weights, k=1):
        if len(population) < k:
            raise ValueError("Population size must be larger than k (number of choices)")
        weights = list(weights)
        positions = range(len(population))
        indices = []
        while True:
            needed = k - len(indices)
            if not needed:
                break
            for i in choices(positions, weights, k=needed):
                if weights[i]:
                    weights[i] = 0.0
                    indices.append(i)
        return [population[i] for i in indices]

    # auxiliary method to create arguments, ensuring that it
    # a) is not quesition-begging (conclusion is also a premise)
    # e) is not attack-reflexive (conclusion is negation of a premise)
    def get_argument2(n_premises_weights):
        # number of premises
        if n_premises_weights is None:
            n_premises_weights = list(np.repeat(1, n_max_premises))

        if variation:
            n_prem = choices(list(range(1,n_max_premises+1)), k=1, weights=n_premises_weights)[0]
            #n_prem = randint(1, n_max_premises)
        else:
            n_prem = n_max_premises

        # Prefer unused sentences:
        if unused_sentences:
            available_sentences = unused_sentences
        else:   # if every sentence has been used, consider all sentences
            available_sentences = set(range(1, n_sentences + 1))

        if connected:
            if not args:    # first argument
                # conclusion: a sentence that is not a principle
                con = choice([s for s in available_sentences if s not in principles])

            else:
                # conclusion: a sentence of an existing argument, which is not a principle

                all_sentences = [s for arg in args for s in arg]
                # set is used to eliminate duplicate sentences
                sentences = list(set(s for s in all_sentences if abs(s) not in principles))
                # prefer sentences that have been used fewer times
                weights = [1/all_sentences.count(s) for s in sentences]

                con = choices(sentences, weights=weights, k=1)[0]

            # update available sentences:
            available_sentences.discard(con)

            sentences = [s for s in available_sentences.union(principles) if s != abs(con)]

            # check whether there are sufficiently many sentences
            if len(sentences) < n_prem:
                # if not consider all sentences
                sentences = [s for s in range(1, n_sentences + 1) if s!=abs(con)]

            # prefer principles as premises in arguments or sentences that have been used fewer times
            weights = [1 if s in principles else 1/sentences.count(s) for s in sentences]

            new_argument = weighted_sample_without_replacement(sentences, weights, k=n_prem) + [con]

        else:
            # ToDo: needs refinement

            # choose different sentences
            new_argument = sample(range(1, n_sentences+1), n_prem+1)

        # randomly apply negation to non-principle sentences
        new_argument = [s if s in principles else s*choice([1, -1]) for s in new_argument]

        return new_argument

    # ToDo-Question: It that what we want? (What do we want?)
    args = []
    loop_counter = 0
    while len(args) < n_arguments:

        new_arg = get_argument2(n_premises_weights)

        # add first argument anyway
        if not args:
            args.append(new_arg)

            # update unused sentences
            for sentence in new_arg:
                unused_sentences.discard(abs(sentence))
        # check whether the constructed argument
        # b) is jointly satisfiable with already existing arguments,
        # c) its premises are not a subset of premises of another argument

        # f) exclude arguments that reshuffle sentences
        # elif not any(set(abs(s) for s in new_arg) == set(abs(t) for t in arg) for arg in args):
        elif is_satisfiable(args + [new_arg], principles=list(principles)):
                # for arguments with more than one premise, check condition c)
                if len(new_arg[:-1]) == 1 or not any(set(new_arg[:-1]).issubset(arg[:-1]) for arg in args):
                    # d) Catch e.g. contrapositions by checking that the addition of the new
                    # argument results in less complete consistent positions
                    if number_of_complete_consistent_positions(args + [new_arg], n_sentences) \
                            < number_of_complete_consistent_positions(args, n_sentences):
                        args.append(new_arg)

                        # update unused sentences
                        for sentence in new_arg:
                            unused_sentences.discard(abs(sentence))

        loop_counter += 1
        if loop_counter == max_loops:
            raise RuntimeWarning("Could not generate enough arguments.\n" +
                                 "The randomly chosen argumentlist (" + str(args) +
                                 ") does not allow to add another argument " +
                                 "within the given maximum number of loop. Try to increase the max_loops or consider" +
                                 " to change the parameters of the desired dialectical structure.")

    # checking whether each sentence or its negation occurs at least once in an argument
    if use_all_sentences:
        if set([abs(s) for arg in args for s in arg]) == set(range(1,n_sentences+1)):
            return args
        else:
            raise RuntimeWarning("Could not generate arguments which include all sentencess. Consider, for instance, "
                                 "increasing the desired number of arguments. ")
    else:
        return args

# todo: rename
def create_random_argument_list(n_arguments_min: int, n_arguments_max: int,
                                n_sentences: int, n_premises_max: int,
                                n_premises_weights=None,
                                ) -> List[List[int]]:
    """A random list of arguments.

    A convenience function that uses :py:func:`create_random_arguments` to create a random list of arguments.
    Instead of specifying a fixed number of desired arguments the number of arguments will be a random
    number between `n_arguments_min` and ` n_arguments_max`.

    """
    n_arguments = randint(n_arguments_min, n_arguments_max)
    args = create_random_arguments(n_sentences=n_sentences, n_arguments=n_arguments,
                                   n_max_premises=n_premises_max, max_loops=1000,
                                   n_premises_weights=n_premises_weights)

    return args


def random_dialectical_structures(n_dialectical_structure,
                                  n_arguments_min,
                                  n_arguments_max,
                                  n_sentences,
                                  n_premises_max,
                                  module_name='rethon.model',
                                  class_name='DAGDialecticalStructure') -> Iterator[DialecticalStructure]:
    """A random list of dialectical structures.

    A convenience function that uses :py:func:`create_random_argument_list` to create an instance of
    :py:class:`basics.DialecticalStructure`.

    Returns:
        A random dialectical structure. The implementing class can be specified via the method's arguments.

    """
    for i in range(n_dialectical_structure):
        args = create_random_argument_list(n_arguments_min, n_arguments_max, n_sentences, n_premises_max)
        ds_class_ = getattr(importlib.import_module(module_name), class_name)
        yield ds_class_.from_arguments(args, n_sentences)

# def random_positions(n_positions, n_sentences,
#                      position_module_name='rethon.model',
#                      position_class_name='StandardPosition') -> Iterator[Position]:
#     """List of random positions.
#
#     Convenience method that uses :py:func:`random_position_as_set` to generate a list of positions. The implementing
#     class can be specified via arguments.
#     """
#     positions_as_set = [random_position_as_set(n_sentences, allow_empty_position=False) for i in range(n_positions)]
#     for pos in positions_as_set:
#         pos_class_ = getattr(importlib.import_module(position_module_name), position_class_name)
#         yield pos_class_.from_set(pos, n_sentences)


class TauJSONEncoder(JSONEncoder):

    def __init__(self, serialize_implementation=False, **kwargs):
        super(TauJSONEncoder, self).__init__(**kwargs)
        self.serialize_implementation = serialize_implementation


    def default(self, o):
        """ An implementation of :py:func:`JSONEncoder.default`.

        This implementation handles the serialization of :class:`Position` and
        :class:`DialecticalStructure` instances.

        """
        if isinstance(o, Position):
            json_dict = {'n_unnegated_sentence_pool': o.sentence_pool(), 'position': list(o.as_set())}
            if self.serialize_implementation:
                json_dict['module_name'] = o.__module__
                json_dict['class_name'] = type(o).__name__
            return json_dict

        if isinstance(o, DialecticalStructure):
            json_dict = {'arguments': o.get_arguments(),
                         'tau_name': o.get_name(),
                         'n_unnegated_sentence_pool': o.sentence_pool().size()
                         }
            if self.serialize_implementation:
                json_dict['module_name'] = o.__module__
                json_dict['class_name'] = type(o).__name__
            return json_dict

        if isinstance(o, np.int64):
            return o.item()
        if isinstance(o, np.int32):
            return o.item()
        if isinstance(o, np.float32):
            return o.item()
        if isinstance(o, set):
            return list(o)

        return JSONEncoder.default(self, o)


def tau_decoder(json_obj,
                use_json_specified_type=False,
                position_module='tau',
                position_class='BitarrayPosition',
                dialectical_structure_module='tau',
                dialectical_structure_class='BitarrayDialecticalStructure'):
    """ Object hook for :py:func:`json.loads` and :py:func:`json.load`.


    :param use_json_specified_type: If :code:`True` the method uses the implementation details
            (modulename and classname) that are specified in the json string, if there are any. Otherwise,
            the method uses implementation details as specified the other given parameters.

    """
    if 'position' in json_obj and 'n_unnegated_sentence_pool' in json_obj:
        if use_json_specified_type and 'module_name' in json_obj and 'class_name' in json_obj:
            position_class_ = getattr(importlib.import_module(json_obj['module_name']),
                                      json_obj['class_name'])
        else:
            position_class_ = getattr(importlib.import_module(position_module),
                                      position_class)
        return position_class_.from_set(json_obj['position'], json_obj['n_unnegated_sentence_pool'])
    if 'arguments' in json_obj and 'n_unnegated_sentence_pool' in json_obj:
        if use_json_specified_type and 'module_name' in json_obj and 'class_name' in json_obj:
            ds_class_ = getattr(importlib.import_module(json_obj['module_name']),
                                json_obj['class_name'])
        else:
            ds_class_ = getattr(importlib.import_module(dialectical_structure_module),
                                dialectical_structure_class)
        if 'tau_name' in json_obj:
            tau_name = json_obj['tau_name']
        else:
            tau_name = None
        return ds_class_.from_arguments(json_obj['arguments'], json_obj['n_unnegated_sentence_pool'],
                                        tau_name)

    return json_obj

def tau_dumps(re_object, cls=TauJSONEncoder, serialize_implementation=False, **kwargs):
    """Getting an object as JSON-String.

    This is a convenient method that calls :py:func:`json.dumps` with :class:`TauJSONEncoder` as
    its default encoder, which will handle the JSON serialization of :class:`Position` and
    :class:`DialecticalStructure`.

    **kwargs will be given to :py:func:`json.dumps`

    :param serialize_implementation: If :code:`True` implementation details (modulename and classname) will
            be serialized.
    :return: The object as a JSON string.

    """
    return dumps(re_object, cls=cls, serialize_implementation=serialize_implementation, **kwargs)


def tau_dump(re_object, fp, cls=TauJSONEncoder, serialize_implementation=False, **kwargs):
    """Saving an object as JSON-String in a file.

    This is a convenient method that calls :py:func:`json.dump` with :class:`TauJSONEncoder` as
    its default encoder, which will handle the JSON serialization of :class:`Position` and
    :class:`DialecticalStructure` instances.

    **kwargs will be given to :py:func:`json.dumps`

    :param serialize_implementation: If :code:`True` implementation details (modulename and classname) will
            be serialized.
    :return: The object as a JSON string.

    """
    dump(re_object, fp, cls=cls, serialize_implementation=serialize_implementation, **kwargs)


def tau_loads(json_obj,
              use_json_specified_type=False,
              position_module='tau',
              position_class='NumpyPosition',
              dialectical_structure_module='tau',
              dialectical_structure_class='BDDNumpyDialecticalStructure'):
    """Loading an object from a JSON string.

    This is a convenient method that calls :py:func:`json.loads` and uses :py:func:`tau_decoder` as object hook
    to handle the instantiation of :class:`Position` and :class:`DialecticalStructure` objects. Desired
    implementation details can be given by parameter values (see :py:func:`tau_decoder`).

    .. note::

            Per default positions will be instantiated as :class:`NumpyPosition` and dialectical structures
            as :class:`BDDNumpyDialecticalStructure` (to avoid long instantiation times).
    """
    return loads(json_obj, object_hook=lambda x: tau_decoder(json_obj = x,
                                                             use_json_specified_type = use_json_specified_type,
                                                             position_module=position_module,
                                                             position_class=position_class,
                                                             dialectical_structure_module=dialectical_structure_module,
                                                             dialectical_structure_class=dialectical_structure_class))


def tau_load(fp,
            use_json_specified_type=False,
            position_module='tau',
            position_class='NumpyPosition',
            dialectical_structure_module='tau',
            dialectical_structure_class='BDDNumpyDialecticalStructure'):
    """Loading an object from a JSON file.

    This is a convenient method that calls :py:func:`json.load` and uses :py:func:`tau_decoder` as object hook
    to handle the instantiation of :class:`Position` and :class:`DialecticalStructure` objects. Desired
    implementation details can be given by parameter values (see :py:func:`tau_decoder`).

    .. note::

            Per default positions will be instantiated as :class:`NumpyPosition` and dialectical structures
            as :class:`BDDNumpyDialecticalStructure` (to avoid long instantiation times).
    """
    return load(fp, object_hook=lambda x: tau_decoder(json_obj = x,
                                                      use_json_specified_type = use_json_specified_type,
                                                      position_module=position_module,
                                                      position_class=position_class,
                                                      dialectical_structure_module=dialectical_structure_module,
                                                      dialectical_structure_class=dialectical_structure_class))

