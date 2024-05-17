for DATA in CoraGraphDataset CiteseerGraphDataset PubmedGraphDataset; do
  for LAYER in GCN GAT BRONX; do
    for STRATEGY in structural; do
      DATA=$DATA LAYER=$LAYER STRATEGY=$STRATEGY SPLIT=-1 bsub < tune.sh
    done
  done
done