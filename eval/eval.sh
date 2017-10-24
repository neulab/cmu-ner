#!/bin/bash
for file in $(ls $1)
do
    echo evaluating $file
    ./conlleval < $1$file
done
