## Important params:
* learning_rate
* encoder: n_layers, d_model, n_heads
* spec_aug
* fuse_loss_wer


## Experiment 1: 
* Dataset: VIVOS 8K
* Model: Fast-Conformer Transducer BPE
* Epoches: 100
![alt text](image.png)


## Experiemnt 2:
* Dataset: VIVOS 16K
* Model: Fast-Conformer Transducer BPE
* Epoches: 100

## Experiemnt 3:
* Dataset: VIVOS 16K + 8K
* Model: Fast-Conformer Transducer BPE
* Epoches: 100
* bucketing_strategy: "fully_randomized"

## Experiment 4:
* Dataset: VIVOS 16K
* Model: Fast-Conformer Hybrid Transducer CTC BPE
* Epoches: 100

## Experiemnt 5:
* Dataset: VIVOS 16K + 8K
* Model: Fast-Conformer Transducer BPE
* Epoches: 100
* bucketing_strategy: "synced_randomized"