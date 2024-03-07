.. _tau-docs-label:


Theodias
========


A Python package for dialectical structures.

The `theodias` package provides different classes and methods to apply the theory of
dialectical structures (as introduced in [Betz2010]_ and [Betz2013]_).

Installation
------------

With :code:`pip` (👉 |theodias_pypi|)

.. code-block:: bash

    pip install theodias

From the source code:

You can install the package locally, by

* first git-cloning the repository:

  * :code:`git clone git@github.com:re-models/theodias.git`

* and then installing the package by running ':code:`pip install -e .`' from the local directory of
  the package (e.g. :code:`local-path-to-repository/tau`) that contains the setup file :code:`setup.py`.
  (The :code:`-e`-option will install the package in the editable mode, allowing you to
  change the source code.)

.. note:: The package requires a python version >= 3.8 and depends on the
    packages `bitarray <https://pypi.org/project/bitarray/>`_, `numba <https://numba.pydata.org/>`_ and `PySat <https://github.com/pysathq/pysat>`_
    (which will be installed by pip automatically).


Documentation
-------------

A Jupyter notebook provides step-by-step instructions of using
the :code:`theodias` package. Further details can be found in the :ref:`API documentation <tau-api-docs-label>`.

.. toctree::
    :hidden:

    Tutorials <tutorials/theodias-tutorials>
    API documentation<api-docs/api>

Logging
-------

:code:`theodias` uses a logger named 'tau' but does not add any logging handlers. Accordingly, logs will be printed to
:code:`sys.stderr` if the application using :code:`theodias` does not specify any other logging configuration
(see `<https://docs.python.org/3/howto/logging.html#configuring-logging-for-a-library>`_).



Licence
-------

|licence|

References
----------

.. [Betz2010] Betz, G. 2010, |betz_2010|, Frankfurt a.M.: Klostermann.

.. [Betz2013] Betz, G. 2013, |betz_2013|, Synthese Library, Dordrecht: Springer 2013.


.. |betz_2010| raw:: html

   <a href="https://doi.org/10.5771/9783465136293" target="_blank">Theorie dialektischer Strukturen</a>

.. |betz_2013| raw:: html

   <a href="https://www.springer.com/de/book/9789400745988" target="_blank">Debate Dynamics: How Controversy Improves Our Beliefs</a>

.. |theodias_pypi| raw:: html

   <a href="https://pypi.org/project/theodias/" target="_blank">https://pypi.org/project/theodias/</a>

.. |licence| raw:: html

   <a href="https://github.com/re-models/theodias?tab=MIT-1-ov-file#readme" target="_blank">MIT Licence</a>