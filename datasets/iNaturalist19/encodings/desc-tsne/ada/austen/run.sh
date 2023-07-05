#!/bin/bash

for ((n_components = 10; n_components <= 1010; n_components += 10)); do
	for ((perplexity = 5; perplexity <= 50; perplexity += 5)); do
		output_file="./datasets/iNaturalist19/encodings/desc-tsne/ada/austen/n_components${n_components}-perplexity${perplexity}.npy"
		if [ ! -f "$output_file" ]; then
			echo "Running with n_components=$n_components and perplexity=$perplexity"
			python encoders/desc-tsne.py \
				--dataset iNaturalist19 \
				--writer austen \
				--embedder ada \
				--n_components $n_components \
				--perplexity $perplexity \
				--random_state 42 \
				--n_jobs 4
		else
			echo "Skipping n_components=$n_components and perplexity=$perplexity as the output file already exists."
		fi
	done
done
