ipy_pep8
========

Simple tool to check PEP8 violations in a IPython notebook

Example Usage
=============

Normally I run

    python pep8_check.pyc -f -o test.py ~/notebooks

to check all notebooks in the ~/notebooks dir. Saving the generated
output to e.g. test.py can be useful to inspect the checked code in
your own editor instead of the notebook itself. So the line numbers
given by pep8 actually make sense.


Ideas
=====

Add interactive editing of flagged code on a chunk basis.
