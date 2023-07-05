#!/bin/bash

beta=0
increment=0.05
while (($(echo "$beta < 1" | bc -l))); do
	echo "Running with beta=$beta"
	python encoders/b3p.py \
		--dataset CIFAR100 \
		--beta $beta
	beta=$(echo "$beta + $increment" | bc -l)
done
