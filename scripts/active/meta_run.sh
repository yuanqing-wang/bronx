for layer in GCN GAT; do
for data in ESOL FreeSolv Lipophilicity; do
for strategy in structural parametric; do
for acquisition in expected_improvement probability_of_improvement; do

LAYER=$layer DATA=$data STRATEGY=$strategy ACQUISITION=$acquisition bsub < run.sh

done
done
done
done