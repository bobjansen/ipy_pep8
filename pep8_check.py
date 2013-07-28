"""
Helper utility to PEP8 check Python code in IPython notebooks.
"""
import json
import os
import subprocess
import sys
import tempfile
from optparse import OptionParser


TEMPLATE_IPYTHON_PRELUDE = """import numpy as np
"""


def find_notebooks(start_directory, extension="ipynb"):
    """
    Finds all notebooks in the start directory.
    """
    notebooks = []
    for root, _, files in os.walk(start_directory):
        notebooks = notebooks + [os.path.join(root, notebook)
                                 for notebook in files
                                 if notebook.endswith(extension)]
    return notebooks


def read_code_from_notebook(notebook):
    """
    Read the source code in a IPython notebook.
    """
    with open(notebook, 'r') as handle:
        text = handle.read()
        data = json.loads(text)
        cells = (cell['input']
                 for cell in data['worksheets'][0]['cells']
                 if cell['cell_type'] == u'code')

        return cells


def check_code(cells, options):
    """
    Check the code in the cells
    """
    temp = tempfile.NamedTemporaryFile()

    try:
        if options.filename is not None:
            handle = open(options.filename, 'w')

        code = ""
        try:
            # Gather all code in a temp file together with a pylab prelude
            code = TEMPLATE_IPYTHON_PRELUDE
            for cell in cells:
                code += "".join(cell) + "\n\n\n"
            temp.write(code)
            temp.flush()
            if options.filename is not None and handle:
                handle.write(code)

            process = subprocess.Popen(['pep8', temp.name],
                                       stdout=subprocess.PIPE)
            result, _ = process.communicate()
            print result
            if result == "":
                success = True
            else:
                success = False
        finally:
            temp.close()
    finally:
        if options.filename is not None and handle:
            handle.close()

    return success


def check_files(start_directory, options):
    """
    Checks all files under start directory.
    """
    failfast = options.failfast

    success = True
    notebooks = find_notebooks(start_directory)
    for notebook in notebooks:
        cells = read_code_from_notebook(notebook)
        success = check_code(cells, options)
        if failfast and not success:
            break
    return success

if __name__ == "__main__":
    PARSER = OptionParser()
    PARSER.add_option("-o", "--output", dest="filename",
                      help="write combined python code to FILE",
                      metavar="FILE")
    PARSER.add_option("-f", "--fail-fast", dest="failfast", default=False,
                      action="store_true",
                      help="fail when the first notebook with an error\
                              is encountered")
    (OPTION, ARGS) = PARSER.parse_args()

    if len(ARGS) == 0:
        SUCCESSES = list(check_files(".", OPTION))
    else:
        SUCCESSES = [check_files(arg, OPTION) for arg in ARGS]
    if reduce(lambda x, y: x and y, SUCCESSES):
        exit(0)
    else:
        exit(1)
