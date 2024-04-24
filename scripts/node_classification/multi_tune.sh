for DATA in CoraGraphDataset CiteseerGraphDataset PubmedGraphDataset; do
  for LAYER in GIN; do
    for STRATEGY in functional parametric structural; do
      DATA=$DATA LAYER=$LAYER STRATEGY=$STRATEGY bsub < tune.sh
    done
  done
done