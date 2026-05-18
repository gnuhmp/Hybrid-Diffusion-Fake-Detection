$datasets = @('politifact', 'gossipcop')
$encoders = @('sbert', 'bert')
$models = @('bigcn', 'gcnfn', 'gnn', 'gnncl', 'gcn', 'gat')

foreach ($dataset in $datasets) {
    foreach ($encoder in $encoders) {
        foreach ($model in $models) {
            Write-Host "Training $model on $dataset with $encoder..."
            python scripts/train_baseline.py --dataset $dataset --encoder $encoder --model $model
        }
    }
}
