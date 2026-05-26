$datasets = @('politifact', 'gossipcop')
$encoders = @('sbert', 'bert')

foreach ($dataset in $datasets) {
    foreach ($encoder in $encoders) {
        Write-Host "Training HyD-Fake on $dataset with $encoder..."
        python scripts/train_hyd_fake.py --dataset $dataset --encoder $encoder
    }
}