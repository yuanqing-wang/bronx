for DATA in CoraGraphDataset; do
  for LAYER in BRONX; do
    for STRATEGY in node parametric; do
      DATA=$DATA LAYER=$LAYER STRATEGY=$STRATEGY SPLIT=-1 bsub < tune.sh
    done
  done
done