#!/usr/bin/env python
import hug, sys
from gnl.hug import main

@hug.cli()
def print_hello(a: float, b: float):
    print(a * b)


if __name__ == '__main__':
    main(__name__)
