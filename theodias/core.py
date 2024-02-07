from .numpy_implementation import DAGNumpyDialecticalStructure, BDDNumpyDialecticalStructure, NumpyPosition


class StandardPosition(NumpyPosition):
    """
    Class that simply tags :py:class:`NumpyPosition` as the default implementation of :py:class:`Position`
    """
    pass


class DAGDialecticalStructure(DAGNumpyDialecticalStructure):
    """
    Class that simply tags :py:class:`DAGNumpyDialecticalStructure` as the default implementation of
    :py:class:`DialecticalStructure` that is based on a directed acyclic graph. (Preferably used together
    with globally searching RE processes (see :py:class:`StandardGlobalReflectiveEquilibrium`).
    """
    pass


class BDDDialecticalStructure(BDDNumpyDialecticalStructure):
    """
    Class that simply tags :py:class:`BDDNumpyDialecticalStructure` as the default implementation of
    :py:class:`DialecticalStructure` that is based on a binary decision trees. (Preferably used together
    with locally searching RE processes (see :py:class:`StandardLocalReflectiveEquilibrium`).
    """
    pass
