# VocoMorph

A Neural Voice Modulation Vocoder

## run.py

Contains the driver code for the entire project. `run.sh` is a wrapper around `run.py` for convenience.
Optionally, set `$PROJECT_ROOT` env var to the directory of the project, otherwise it will be set automatically to the directory of `run.py`.
You can also set `$DATA_ROOT` if you're storing your data elsewhere, otherwise it will be set automatically as `$PROJECT_ROOT/data`
eg run: `python run.py --mode train --model VocoMorph --dataset timit`

## run.sh

Wrapper around run.py for convenience. It supports DDP training via `torchrun` or regular training (note: this adds the -ddp flag to run.py).

## Training

To train, A directory for the model needs to be created.
You may structure the directory as you like, but you need:

1. `model.py` module with a class matching the name of the model directory.
2. `config.yaml` in that directory

eg: create a MyVocomorphModel in `$PROJECT_ROOT/models/MyVocomorphModel`, and define a class MyVocomorphModel inside `$PROJECT_ROOT/models/MyVocomorphModel/model.py`
`Trainer` will automatically save and load checkpoints from `MyVocomorphModel/checkpoints`.

## Config

This is where config parameters for the model, as well as any other configuration options for the trainer, checkpointer, dataloaders, ...etc are defined.

### Custom classes

The YAML files can be modified to reference both custom and builtin functions/classes for criterions, schedulers & metrics.
You may specify a weight to multiply the value of the loss by when aggregating.

eg:

```yaml
criterions:
  - name: "SISNRLoss"
    weight: 1.0

  - name: "MultiResolutionSTFTLoss"
    weight: 1.0
    params:
      alpha: 1.0
      beta: 0.5
      resolutions: [
          [512, 512, 128], # Moderate frequency resolution, good time resolution
          [1024, 1024, 256], # Good frequency resolution, medium time resolution
          [2048, 2048, 512], # Excellent frequency resolution, poor time resolution
        ]
```

The same goes for effects that are used for augment the data.
First implement the function in `$PROJECT_ROOT/src/modulation/effects.py` with signature: `def apply_my_effect(audio: NDArray, sr: int) -> NDArray:`
Then add it to the list in:

```yaml
data:
  effects: ["apply_my_effect"]
```

### Model summary

`Trainer` supports dynamic model summary generation.
if `dummy_input` key is present in the model config, it will use that blueprint to construct an input to be used in model summary.
Without dummy_input less detailed information about the model will be displayed.

for eg, to pass a tuple of (Tensor(1, torch.long), Tensor(1, 1, 16_000, torch.float32):

```yaml
dummy_input:
  - shape: [1]
    dtype: long
  - shape: [1, 1, 16000]
    dtype: float32
```

To pass a dict:

```yaml
dummy_input:
  a:
    - shape: [1]
      dtype: long
  b:
    - shape: [1, 1, 16000]
      dtype: float32
```

## Inference

For inference, a checkpoint and sample file are needed.
`python run.py --mode infer_sample --sample-file /path/to/sample.wav --model VocoMorph --checkpoint-path /path/to/checkpoint`
