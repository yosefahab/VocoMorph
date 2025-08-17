# VocoMorph

A Neural Voice Modulation Vocoder

## run.py

Contains the driver code for the entire project. `run.sh` is a wrapper around `run.py` for convenience.
Optionally, set $PROJECT_ROOT env var to the directory of the project, otherwise it will be set automatically to the directory of `run.py`.
You can also set $DATA_ROOT if you're storing your data elsewhere, otherwise it will be set automatically as $PROJECT_ROOT/data
eg run: `python run.py --mode train --model VocoMorph --dataset librispeech`

## run.sh

Wrapper around run.py for convenience.

## Training

To train, A directory for the model needs to be created.
You may structure the directory as you like, but you need:

1. `model.py` module with a class matching the name of the model directory.
2. `config.yaml` in that directory

eg: create a MyVocomorphModel in $PROJECT_ROOT/models/MyVocomorphModel, and define a class MyVocomorphModel inside $PROJECT_ROOT/models/MyVocomorphModel/model.py
`Trainer` will automatically save and load checkpoints from `MyVocomorphModel/checkpoints`.

## Config

This is where config parameters for the model, as well as any other configuration options for the trainer, checkpointer, dataloaders, ...etc are defined.

### datalists:

These are csv files that contain the paths to the training samples.

It looks like this:

```csv
ID,effect_id,data_path
wave_id,effect_id,/path/to/.npz
```

**IMPORTANT**
`effect_id` 0 is always the raw waveform. This means that for every other `effect_id` > 0, there has to exist a wave with the same ID with `effect_id` == 0.

### Custom classes

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
First implement the function in $PROJECT_ROOT/src/modulation/effects.py with signature: `def apply_my_effect(audio: NDArray, sr: int) -> NDArray:`
Then add it to the list in:

```yaml
data:
  effects: ["apply_my_effect"]
```

## Inference

For inference, a checkpoint and sample file are needed.
`python run.py --mode infer_sample --sample-file /path/to/sample.wav --model VocoMorph --checkpoint-path /path/to/checkpoint`
