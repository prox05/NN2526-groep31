# NN2526-groep31

## Group Members
- Sabih Syed - 15611175
- Yousri Nait Aicha - 14651440
- Trayvaun Palmer

## Assignment Setup (Food Classification)
This repo now includes a complete baseline pipeline:
- `src/train.py`: trains a ResNet18 classifier on `Data/train_set/...`
- `src/predict.py`: loads the best checkpoint and creates submission CSV (`img_name,label`)

The scripts already handle your nested dataset folders:
- `Data/train_set/train_set/train_set/*.jpg`
- `Data/test_set/test_set/test_set/*.jpg`

## 1) Create environment and install dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 2) Train a baseline model
Fast CPU sanity check:
```bash
python src/train.py --epochs 1 --batch-size 32 --num-workers 2
```

Better run (recommended with GPU):
```bash
python src/train.py --epochs 20 --batch-size 64 --num-workers 4 --pretrained --early-stopping-patience 5
```

Outputs:
- Checkpoint: `artifacts/best_model.pt`
- Training metadata: `artifacts/train_meta.json`

## 3) Generate submission CSV
```bash
python src/predict.py --checkpoint artifacts/best_model.pt --output-csv submissions/submission.csv
```

Output format is exactly what the assignment expects:
```text
img_name,label
test_1.jpg,4
test_2.jpg,2
...
```

## Useful notes
- Labels in training data are `1..80`; training internally converts to `0..79`, then converts back on submission.
- If `--pretrained` cannot download ImageNet weights, training falls back to random initialization automatically.
- The reported metric during training is validation accuracy, which is a good local proxy for leaderboard category accuracy.
- Training now uses stronger augmentations (`RandomResizedCrop`, `ColorJitter`, `RandomRotation`), `CosineAnnealingLR`, and early stopping.

## Suggested next improvements
- K-fold training + model ensembling for better test predictions
- Try larger backbones (e.g., ResNet50, EfficientNet) if GPU budget allows
- Test-time augmentation (TTA) for slightly more robust predictions
