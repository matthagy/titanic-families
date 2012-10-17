#!/bin/bash
# Helper script for generating dot images with components layed out
# in a grid format

dot $1 | gvpack -array$3 | neato -Tpng -n2 -o $2 

