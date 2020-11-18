#!/bin/bash 

DATA_PREFIX=./

./test2 -1 > $DATA_PREFIX/2-base.txt
./test2 0 > $DATA_PREFIX/2-0.txt
./test2 1 > $DATA_PREFIX/2-1.txt
./test2 2 > $DATA_PREFIX/2-2.txt
./test2 3 > $DATA_PREFIX/2-3.txt

./test4 -1 > $DATA_PREFIX/4-base.txt
./test4 0 > $DATA_PREFIX/4-0.txt
./test4 1 > $DATA_PREFIX/4-1.txt
./test4 2 > $DATA_PREFIX/4-2.txt
./test4 3 > $DATA_PREFIX/4-3.txt

./test8 -1 > $DATA_PREFIX/8-base.txt
./test8 0 > $DATA_PREFIX/8-0.txt
./test8 1 > $DATA_PREFIX/8-1.txt
./test8 2 > $DATA_PREFIX/8-2.txt
./test8 3 > $DATA_PREFIX/8-3.txt

./test16 -1 > $DATA_PREFIX/16-base.txt
./test16 0 > $DATA_PREFIX/16-0.txt
./test16 1 > $DATA_PREFIX/16-1.txt
./test16 2 > $DATA_PREFIX/16-2.txt
./test16 3 > $DATA_PREFIX/16-3.txt

./test2s -1 > $DATA_PREFIX/2s-base.txt
./test2s 0 > $DATA_PREFIX/2s-0.txt
./test2s 1 > $DATA_PREFIX/2s-1.txt
./test2s 2 > $DATA_PREFIX/2s-2.txt
./test2s 3 > $DATA_PREFIX/2s-3.txt

./test4s -1 > $DATA_PREFIX/4s-base.txt
./test4s 0 > $DATA_PREFIX/4s-0.txt
./test4s 1 > $DATA_PREFIX/4s-1.txt
./test4s 2 > $DATA_PREFIX/4s-2.txt
./test4s 3 > $DATA_PREFIX/4s-3.txt

./test8s -1 > $DATA_PREFIX/8s-base.txt
./test8s 0 > $DATA_PREFIX/8s-0.txt
./test8s 1 > $DATA_PREFIX/8s-1.txt
./test8s 2 > $DATA_PREFIX/8s-2.txt
./test8s 3 > $DATA_PREFIX/8s-3.txt

./test16s -1 > $DATA_PREFIX/16s-base.txt
./test16s 0 > $DATA_PREFIX/16s-0.txt
./test16s 1 > $DATA_PREFIX/16s-1.txt
./test16s 2 > $DATA_PREFIX/16s-2.txt
./test16s 3 > $DATA_PREFIX/16s-3.txt