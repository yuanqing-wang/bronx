for DATA in CoraGraphDataset; do
  for LAYER in GCN; do
    for STRATEGY in node; do
      DATA=$DATA LAYER=$LAYER STRATEGY=$STRATEGY SPLIT=-1 bsub < tune.sh
    done
  done
done