"""Useful addons for hug

This module provides a function `main` for running files with hug decorators from the command line like

    test.py <fun> [args...]

"""
import hug, sys
from hug.api import API


def main():
    api = API(sys.modules[__name__])

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
