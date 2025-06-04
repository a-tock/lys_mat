.. lys_mat documentation master file, created by
   sphinx-quickstart on Wed Jul 24 18:36:57 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

lys_mat documentation
===================================

*lys_mat* is a Python-based library for crystal structures.
Source code of *lys_mat* is opened in GitHub (https://github.com/a-tock/lys_mat).

*lys_mat* can be used as an extension to *lys* (https://lys-devel.github.io/lys/index.html), a multi-dimensional data analysis and visualization platform. However, it can also be used as standalone library if you do not need GUI.

To use *lys_mat*, go to :doc:`install` and try :doc:`tutorial`.

Characteristics:

- It is designed to be used in conjunction with *sympy* (https://www.sympy.org/en/index.html), a Python library for symbolic mathematics.
- In lys_mat, crystal structure parameters can be treated as sympy objects.
- By combining this with simulation, it becomes possible to freely fit experimental data, and it is easy to determine crystal structures from experimental results.

Future vision

- Creation of a GUI tool that can visualize and edit crystal structures

This library is still under developement.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   install
   tutorial
   api
   contributing
