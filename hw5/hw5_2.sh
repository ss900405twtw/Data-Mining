#!/bin/bash 
$1/svm-train -c 64 -g 1 $2
$1/svm-predict $3 news.tr.model $4
