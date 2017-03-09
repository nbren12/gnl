"""Useful addons for hug

This module provides a function `main` for running files with hug decorators from the command line like

    test.py <fun> [args...]

This is an exmaple of a file implementing this function::

    #!/usr/bin/env python
    import hug, sys
    from gnl.hug import main

    @hug.cli()
    def print_hello(a: float, b: float):
        print(a * b)


    if __name__ == '__main__':
        main(__name__)
"""
import hug, sys
from hug.api import API


def main(name):
    """Run hug command line interface


    Examples
    --------
    >>> main(__name__)
    """
    api = API(sys.modules[name])

    try:
        command = sys.argv[1]
    except IndexError:
        print(str(api.cli))
        sys.exit(1)

    if command not in api.cli.commands:
        print(str(api.cli))
        sys.exit()
    else:
        sys.argv.pop(1)
        api.cli.commands[command]()
