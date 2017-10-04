#!/bin/bash
nvcc -cubin -arch=sm_35 kernel2.cu
rm kernel2.sass
KeplerAs.pl -e kernel2.cubin >> kernel2.sass
