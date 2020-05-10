#!/bin/bash 
$1/svm-train -s 0 -t 0 $2
$1/svm-predict $3 iris.tr.model $4
