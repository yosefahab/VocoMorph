# VocoMorph

## Config
The YAML files can be modified to reference both custom and builtin functions/classes for criterions, schedulers & metrics.

eg:
```yaml
criterion:
    name: [ "STFTLoss", "MyLoss" ]
    STFTLoss:
        n_fft: 1024
        hop_length: 256
        win_length: 1024
    MyLoss:
        param1: 1
        list_param: [1, 2, 3]
```


The same goes for effects that are used for augment the data.
First implement the function in src/modulation/effects.py with signature: `def apply_my_effect(audio: np.ndarray, sr: int) -> np.ndarray:`
Then add it to the list in:
```yaml
data:
    effects: [ "apply_my_effect" ]
```

