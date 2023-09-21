# Hyperparameters for CIFAR100 dataset projections

python projectors/umap-projector.py \
	--dataset CIFAR100 \
	--n_neighbors 50 \
	--min_dist 2 \
	--spread 3 \
	--regex "barz-denzler.npy"

python projectors/umap-projector.py \
	--dataset CIFAR100 \
	--n_neighbors 50 \
	--min_dist 2 \
	--spread 3 \
	--regex "*beta*.npy"

python projectors/umap-projector.py \
	--dataset CIFAR100 \
	--n_neighbors 20 \
	--min_dist .1 \
	--spread 1 \
	--regex "*desc*.npy"

python projectors/umap-projector.py \
	--dataset CIFAR100 \
	--n_neighbors 20 \
	--min_dist .1 \
	--spread 1 \
	--regex "*d*00.npy"
