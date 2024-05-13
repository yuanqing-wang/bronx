for DATA in ESOL; do
  for LAYER in BRONX; do
    for STRATEGY in node parametric; do
      DATA=$DATA LAYER=$LAYER STRATEGY=$STRATEGY bsub < tune.sh
    done
  done
done