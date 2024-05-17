for layer in GCN GAT; do
for data in Cora Citeseer Pubmed; do
for strategy in Structural Parametric; do
for percentage in 0.1; do

LAYER=$layer DATA=$data STRATEGY=$strategy PERCENTAGE=$percentage bsub < run.sh

done
done
done
done