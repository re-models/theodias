
# Todo: Add and test checking dependencies

from .core import (
    StandardPosition,
    DAGDialecticalStructure,
    BDDDialecticalStructure
)
from .base import (
    Position,
    DialecticalStructure,
)
from .set_implementation import (
    SetBasedPosition,
    DAGSetBasedDialecticalStructure
)
from .bitarray_implementation import (
    BitarrayPosition,
    DAGBitarrayDialecticalStructure
)
from .numpy_implementation import (
    NumpyPosition,
    BDDNumpyDialecticalStructure,
    DAGNumpyDialecticalStructure
)
# from .util import (
#     is_satisfiable,
#     inferential_density,
#     get_principles,
#     number_of_complete_consistent_positions,
#     arg_to_cnf,
#     args2cnf,
#     write_as_tex,
#     write_as_dot,
#     save_dialectical_structure,
#     load_dialectical_structure,
#     random_position_as_set,
#     create_random_arguments2,
#     create_random_argument_list,
#     random_dialectical_structures,
#     random_positions,
#     TauJSONEncoder,
#     tau_decoder,
#     tau_dump,
#     tau_dumps,
#     tau_load,
#     tau_loads
# )

# Use __all__ to let type checkers know what is part of the public API.
__all__ = [
    "StandardPosition",
    "DAGDialecticalStructure",
    "BDDDialecticalStructure",
    "Position",
    "DialecticalStructure",
    "SetBasedPosition",
    "DAGSetBasedDialecticalStructure",
    "BitarrayPosition",
    "DAGBitarrayDialecticalStructure",
    "NumpyPosition",
    "BDDNumpyDialecticalStructure",
    "DAGNumpyDialecticalStructure",
    # "is_satisfiable",
    # "inferential_density",
    # "get_principles",
    # "number_of_complete_consistent_positions",
    # "arg_to_cnf",
    # "args2cnf",
    # "write_as_tex",
    # "write_as_dot",
    # "save_dialectical_structure",
    # "load_dialectical_structure",
    # "random_position_as_set",
    # "create_random_arguments2",
    # "create_random_argument_list",
    # "random_dialectical_structures",
    # "random_positions",
    # "TauJSONEncoder",
    # "tau_decoder",
    # "tau_dump",
    # "tau_dumps",
    # "tau_load",
    # "tau_loads"
]
