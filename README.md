# Art Classifier

Image classification model that identifies artworks by 50 famous artists using EfficientNet.


## Dataset


```
data/
└── images/
    └── images/
        ├── Alfred_Sisley/
        ├── Amedeo_Modigliani/
        └── ...
```

### Serve (Gradio)
```bash
cd src
python serve.py
```

## Project Structure

```
├── data/                   # Dataset (download separately)
├── outputs/
│   ├── models/             # Saved model checkpoints
│   └── plots/              # Evaluation visualizations
├── src/
│   ├── config.py           # Hyperparameters and paths
│   ├── dataset.py          # Data loading utilities
│   ├── model.py            # EfficientNet architecture
│   ├── train.py            # Training script
│   ├── eval.py             # Evaluation with charts
│   └── serve.py            # Gradio web interface
└── requirements.txt
```
# deep_learning_private


