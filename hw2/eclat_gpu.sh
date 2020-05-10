#!/bin/bash
nvcc gpu.cu -o gpu.out --std=c++11
./gpu.out $1 $2 $3

