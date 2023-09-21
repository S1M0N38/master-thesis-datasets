# Hyperparameters for iNaturalist19 dataset projections

python projectors/umap-projector.py \
	--dataset iNaturalist19 \
	--n_neighbors 505 \
	--min_dist 11 \
	--spread 12 \
	--regex "barz-denzler.npy"

python projectors/umap-projector.py \
	--dataset iNaturalist19 \
	--n_neighbors 505 \
	--min_dist 11 \
	--spread 12 \
	--regex "*beta*.npy"

python projectors/umap-projector.py \
	--dataset iNaturalist19 \
	--n_neighbors 505 \
	--min_dist 5 \
	--spread 12 \
	--regex "*desc*.npy"

python projectors/umap-projector.py \
	--dataset iNaturalist19 \
	--n_neighbors 505 \
	--min_dist 5 \
	--spread 12 \
	--regex "*d*00.npy"
