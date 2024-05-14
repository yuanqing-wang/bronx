for DATA in CiteseerGraphDataset; do
  for LAYER in BRONX GCN GAT; do
    for STRATEGY in node parametric; do
      DATA=$DATA LAYER=$LAYER STRATEGY=$STRATEGY SPLIT=-1 sbatch tune.sbatch
    done
  done
done