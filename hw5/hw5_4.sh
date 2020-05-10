#!/bin/bash 
python3 income.py l $2 $3
$1/svm-scale income.tr > income_scale.tr
$1/svm-scale income.te > income_scale.te
$1/svm-train -c 24 -g 0.005 income_scale.tr
$1/svm-predict income_scale.te income_scale.tr.model $4
