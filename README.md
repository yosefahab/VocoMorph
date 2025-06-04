# VocoMorph

A Neural Voice Modulation Vocoder

## run.py

Contains the driver code for the entire project. `run.sh` is a wrapper around `run.py` for convenience.
eg run: `python run.py --engine-mode train --model-name VocoMorph`

## Training

To train, you need to create a directory for your model.
You may structure the directory as you like, but you need:

1. `model.py` module with a class matching the name of the model directory.
2. `config.yaml` in that directory

eg: create a MyVocomorphModel in $PROJECT_ROOT/models/MyVocomorphModel, and define a class MyVocomorphModel inside $PROJECT_ROOT/models/MyVocomorphModel/model.py

### Config

This is where config parameters for your model and its submodules are defined, as well as any other configuration options for the trainer, checkpointer, dataloaders, ...etc are defined.

#### Custom classes

The YAML files can be modified to reference both custom and builtin functions/classes for criterions, schedulers & metrics.

eg:

```yaml
criterion:
  name: ["STFTLoss", "MyLoss"]
  STFTLoss:
    n_fft: 1024
    hop_length: 256
    win_length: 1024
  MyLoss:
    param1: 1
    list_param: [1, 2, 3]
```

The same goes for effects that are used for augment the data.
First implement the function in $PROJECT_ROOT/src/modulation/effects.py with signature: `def apply_my_effect(audio: np.ndarray, sr: int) -> np.ndarray:`
Then add it to the list in:

```yaml
data:
  effects: ["apply_my_effect"]
```

## Inference

For inference, a checkpoint and sample file are needed.
`python run.py --mode infer_sample --sample-file /path/to/sample.wav --model-name VocoMorph --checkpoint-path /path/to/checkpoint`
