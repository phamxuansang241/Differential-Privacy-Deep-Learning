# Deep Learning with Differential Privacy

Simple implementation of **Deep Learning (DL)** with **Differential Privacy (DP)**.

## Requirements
- torch 1.12.1
- functorch 0.2.1
- numpy 1.16.2
- opacus 1.3.0


## Usage
1. Execute run_model.py -cf dp_sgd_config.json

### Model parameters
```python
{
    "data_name": "mnist",
    "epochs": 100,
    "batch_size": 128, 
    "lr": 0.001, 
    "epsilon": 6, 
    "delta": 1e-05,
    "clipping_norm": 16.0, 
    "q": 0.05
}
```

## Reference
[1] 

