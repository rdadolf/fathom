#!/bin/bash
die() { echo "ERROR: $@"; exit 1; }
pylint -rn fathom/ || die 'Lint check failed for fathom/'
pylint -rn test/ || die 'Lint check failed for test/'
nosetests -v test/ || die 'Regression tests failed'
