#!/bin/bash
nvcc cpu.cu -o cpu.out --std=c++11
./cpu.out $1 $2 $3
