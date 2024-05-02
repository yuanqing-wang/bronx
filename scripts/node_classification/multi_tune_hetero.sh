for DATA in CornellDataset; do
  for LAYER in GCN; do
    for SPLIT in 0; do
      for STRATEGY in node parametric; do
        DATA=$DATA LAYER=$LAYER STRATEGY=$STRATEGY SPLIT=$SPLIT bsub < tune.sh
        done  
    done
  done
done