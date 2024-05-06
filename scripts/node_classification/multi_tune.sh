for DATA in CoraGraphDataset CiteseerGraphDataset PubmedGraphDataset; do
  for LAYER in GCN GAT; do
    for STRATEGY in node parametric; do
      DATA=$DATA LAYER=$LAYER STRATEGY=$STRATEGY SPLIT=-1 bsub < tune.sh
    done
  done
done