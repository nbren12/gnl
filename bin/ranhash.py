#!/usr/bin/env python
"""Generate random hash

Usage:
  ranhash.py [N]
"""
from random import randrange
from docopt import docopt

hex = '0123456789abcdef'

args = docopt(__doc__)

N = args['N']
if N is None:
    N = 40
length = int(N)

print(''.join(hex[randrange(0, len(hex))] for _ in range(length)))

