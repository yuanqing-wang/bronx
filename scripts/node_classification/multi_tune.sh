for DATA in CoraGraphDataset CiteseerGraphDataset PubmedGraphDataset; do
  for LAYER in GCNII; do
    # for STRATEGY in functional parametric structural; do
    for STRATEGY in parametric; do
      DATA=$DATA LAYER=$LAYER STRATEGY=$STRATEGY bsub < tune.sh
    done
  done
done