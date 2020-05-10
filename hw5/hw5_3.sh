#!/bin/bash 
python3 abalone.py l $2 $3
$1/svm-train -c 512 -g 0.5 abalone.tr
$1/svm-predict abalone.te abalone.tr.model $4
