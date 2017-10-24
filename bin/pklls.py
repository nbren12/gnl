#!/usr/bin/env python
import sys
from sklearn.externals import joblib

filename = sys.argv[1]

d = joblib.load(filename)

for key in d:
    print(key + ": "+  str(d[key])[:80])
