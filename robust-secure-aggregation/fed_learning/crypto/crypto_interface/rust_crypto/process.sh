#!/bin/bash
FILES=../../../../benchmarks/*
OUT=out.txt
for f in $FILES
do
#  echo "Processing $f file..."
  # take action on each file. $f store current file name
  AVG=$(perl -lane '$a+=$_ for(@F);$f+=scalar(@F);END{print "".$a/$f}' "$f")
  echo "$f $AVG"
#  cat $f >> $OUT
done