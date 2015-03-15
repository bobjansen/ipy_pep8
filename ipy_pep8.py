"""
Helper utility to PEP8 check and fix Python code in IPython notebooks.
"""
import json
import collections
import os
import pep8
import autopep8
import codecs
import encodings
from optparse import OptionParser


TEMPLATE_IPYTHON_PRELUDE = """import numpy as np
"""


def find_notebooks(start_directory, extension="ipynb"):
    """
    Finds all notebooks in the start directory.
    """
    notebooks = []
    for root, _, files in os.walk(start_directory):
        notebooks += [os.path.join(root, notebook) for notebook in files
                      if notebook.endswith(extension)]
    return notebooks


def read_notebook(notebook, encoding="utf-8"):
    """
    Read the source code in a IPython notebook.
    """
    with codecs.open(notebook, 'r', encoding=encoding) as handle:
        # The second parameter enables preserving the order of the JSON fields
        # to minimize changes in the file
        data = json.load(handle, object_pairs_hook=collections.OrderedDict)
        if data[u"nbformat"] not in (3, 4):
            raise NotImplementedError("Only nbformats 3 and 4 are supported.")

    return data

def get_code_cells(notebook_data, notebook_name='This notebook'):
    """
    Get code cells from notebook.
    """
    try:
        # Refactoring this might be a good idea but only 2 cases now...
        if notebook_data[u"nbformat"] == 4:
            code_cells = ((cell_num, cell['source']) for cell_num, cell
                          in enumerate(notebook_data['cells'])
                          if cell['cell_type'] == u'code')
        elif notebook_data[u"nbformat"] == 3:
            code_cells = ((cell_num, cell['input']) for cell_num, cell
                          in enumerate(notebook_data['worksheets'][0]['cells'])
                          if cell['cell_type'] == u'code')
        else:
            # This shouldn't happen in normal usage but could happen if
            # ipy_pep8 is used as a library.
            raise NotImplementedError("Only nbformats 3 and 4 are supported")
        return code_cells
    except IndexError:
        print("%s is not a valid notebook." % (notebook_name, ))
        return []
    except NotImplementedError:
        raise


def check_code(cells):
    """
    Check the code in the cells with pep8, returning True iff there are no errors.
    """
    style = pep8.StyleGuide(parse_argv=False, config_file=True)

    num_errors = 0

    for cell_num, cell_lines in cells:
        print('checking code cell %d' % (cell_num, ))
        #cell_lines = [TEMPLATE_IPYTHON_PRELUDE] + cell_lines + ["\n\n\n"]
        checker = pep8.Checker(lines=cell_lines, options=style.options)
        num_errors += checker.check_all()

    if num_errors == 0:
        return True
    else:
        return False


def fix_code(cells, options):
    """
    Returns code lines fixed with autopep8.
    """
    autopep8_cmdline = '- ' + options.autopep8_options # Required filename arg
    autopep8_options, autopep8_args = autopep8.parse_args(autopep8_cmdline.split())
    fixed_cells = []
    for cell_num, cell_lines in cells:
        fixed_code = autopep8.fix_lines(cell_lines, autopep8_options)
        fixed_lines = fixed_code.splitlines(True)

        if options.no_newline and fixed_lines:
            # Remove the single newline at end of 'file' added by autopep8 to
            # each cell.
            fixed_lines[-1] = fixed_lines[-1][:-1]

            if options.end_semicolon and not fixed_lines[-1].endswith('?'):
                fixed_lines[-1] += ';'

        fixed_cells.append((cell_num, fixed_lines))
    return fixed_cells


def update_code_cells(notebook_data, code_cells):
    """
    Update the code cells.
    """
    # The root of the cells is different between versions.
    if notebook_data[u"nbformat"] == 4:
        cells = notebook_data['cells']
        source_key = 'source'
    elif notebook_data[u"nbformat"] == 3:
        cells = notebook_data['worksheets'][0]['cells']
        source_key = 'input'
    else:
        raise NotImplementedError("Only nbformats 3 and 4 are supported.")

    # Update the source cells.
    for cell_num, cell_lines in code_cells:
        cells[cell_num][source_key] = cell_lines

def write_notebook(notebook, data, encoding="utf-8"):
    """
    Write the notebook.
    """
    with codecs.open(notebook, 'w', encoding=encoding) as handle:
        json.dump(data, handle, indent=1, separators=(',', ': '),
                  ensure_ascii=False)
        handle.write("\n")


def process_files(start_directory, options):
    """
    Checks all files under start directory.
    """
    failfast = options.failfast

    results = []

    notebooks = find_notebooks(start_directory)
    for notebook in notebooks:
        print('Processing notebook %s' % (notebook,))
        try:
            notebook_data = read_notebook(notebook, options.encoding)
            code_cells = get_code_cells(notebook_data)
            if not options.autopep8:
                result = check_code(code_cells)
            else:
                fixed_code_cells = fix_code(code_cells, options)
                update_code_cells(notebook_data, fixed_code_cells)
                write_notebook(notebook, notebook_data, options.encoding)
                result = True # Everything's fixed automatically
            results.append(result)
            if failfast and not result:
                break
        except NotImplementedError as err:
            print "Notebook not supported", err
            print ("Note that IPython versions and format versions do not "
                "necessarily match.\n")
    return results

if __name__ == "__main__":
    PARSER = OptionParser()
    PARSER.add_option("-a", "--autopep8", action="store_true",
                      help="fix notebooks in-place with autopep8")
    PARSER.add_option("-f", "--fail-fast", dest="failfast", action="store_true",
                      help="fail when the first notebook with an error is "
                           "encountered")
    PARSER.add_option("-n", "--no-newline", dest="no_newline", action="store_true",
                      help="Leave no newline at end of cell when running -a. "
                            "This assumes W292 isn't ignored.")
    PARSER.add_option("--end-semicolon", dest="end_semicolon", action="store_true",
                      help="Add a semicolon at the end of each code cell. "
                            "This assumes E703 isn't ignored and applies only "
                            "if -n is chosen.")
    PARSER.add_option('--autopep8-options', dest='autopep8_options', action='store',
                      type='string', default='--ignore=E501',
                      help="(in quotes) passed to autopep8. "
                           "pep8 arguments can be passed via its config file.")
    PARSER.add_option('-e', '--encoding', dest='encoding', action='store',
                      type="string", default="utf-8",
                      help="Specifies the encoding of the original files."
                           "The encoding will be preserved")
    (OPTIONS, ARGS) = PARSER.parse_args()

    encoding = OPTIONS.encoding

    if encodings.search_function(encoding) is None:
        if encoding.startswith("-"):
            print "error: --encoding option requires an argument"
            exit(1)
        else:
            print "Specified encoding not found"
            exit(1)

    if len(ARGS) == 0:
        SUCCESSES = process_files(".", OPTIONS)
    else:
        SUCCESSES = sum((process_files(arg, OPTIONS) for arg in ARGS), [])
    if all(SUCCESSES):
        exit(0)
    else:
        exit(1)

