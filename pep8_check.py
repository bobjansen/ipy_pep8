"""
Helper utility to PEP8 check Python code in IPython notebooks.
"""
import json
import os
import subprocess
import sys
import tempfile


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


def check_code(cells):
    """
    Check the code in the cells
    """
    temp = tempfile.NamedTemporaryFile()
    try:
        # Gather all code in a temp file together with a pylab prelude
        temp.write(TEMPLATE_IPYTHON_PRELUDE)
        for cell in cells:
            temp.write("".join(cell))

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

    return success


def check_files(start_directory):
    """
    Checks all files under start directory.
    """
    notebooks = find_notebooks(start_directory)
    for notebook in notebooks:
        cells = read_code_from_notebook(notebook)
        success = check_code(cells)
        if not success:
            break

if __name__ == "__main__":
    if len(sys.argv) == 1:
        ROOT = "."
    else:
        ROOT = sys.argv[1]
    check_files(ROOT)
