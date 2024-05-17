for DATA in ESOL FreeSolv Lipophilicity; do
  for LAYER in GCN GAT GIN BRONX; do
    for STRATEGY in parametric structural; do
      DATA=$DATA LAYER=$LAYER STRATEGY=$STRATEGY bsub < tune.sh
    done
  done
done