#!/bin/bash

for ((n_components = 10; n_components <= 100; n_components += 10)); do
	for ((perplexity = 5; perplexity <= 50; perplexity += 5)); do
		echo "Running with n_components=$n_components and perplexity=$perplexity"
		python encoders/desc-tsne.py \
			--dataset CIFAR100 \
			--writer austen \
			--embedder ada \
			--n_components $n_components \
			--perplexity $perplexity \
			--random_state 0
	done
done
