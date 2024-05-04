for DATA in ESOL; do
  for LAYER in GCN; do
    for STRATEGY in node; do
      DATA=$DATA LAYER=$LAYER STRATEGY=$STRATEGY bsub < tune.sh
    done
  done
done