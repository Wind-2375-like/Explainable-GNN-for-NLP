# Explainable-GNN for NLP Tasks

We aim to find a new perturbation-based technique for GNN-explainability. Now this repository contains an implementation of [GraphMask](https://arxiv.org/abs/2010.00577) on two NLP tasks: [DialogueGCN](https://arxiv.org/abs/1908.11540.pdf) and [Text GCN](https://arxiv.org/abs/1809.05679). We will introduce more tasks and build a new analyser on all these tasks.

## Requirements

Verify that you have the following dependencies:

* Python 3.6
* PyTorch 1.9.0
* PyTorch Geometric 1.7.2

## Running the Code

For Graphmask on DialogueGCN, just run:

```
python train_IEMOCAP.py
```

For GraphMask on Text GCN, just run:

```
python train_TextGCN.py
```

## Performance

**DialogueGCN:**

|            -            | **Dataset** | **Weighted F1** |
| :---------------------: | :---------: | :-------------: |
|      **Original**       |   IEMOCAP   |     0.6418      |
| **This Implementation** |   IEMOCAP   |     0.6396      |
| **Gated by Graphmask**  |   IEMOCAP   |     0.6139      |

- Divergence ($\frac {|{F1_{Gated}-F1_{Original}}|} {F1_{Original}}$ on test set): 4.02% 
- Sparsity: 38.68% (layer 0: 0 | layer 1: 77.36% )

**Text GCN:**

|            -            | **Dataset** | **F1** | **Acc** |
| :---------------------: | :---------: | :----: | :-----: |
|      **Original**       |     R8      |   -    | 0.9707* |
| **This Implementation** |     R8      | 0.9042 | 0.9730  |
| **Gated by Graphmask**  |     R8      | 0.8687 | 0.9566  |

**NOTE: The result of the official experiment is to repeat the run 10 times, and then take the average of accuracy, but we only ran it once.**

- Divergence ($\frac {1}{2} \times (\frac {|{F1_{Gated}-F1_{Original}}|}{F1_{Original}}+\frac {|{Acc_{Gated}-Acc_{Original}}|}{Acc_{Original}})$ on test set): 2.81%
- Sparsity: 46.11% (layer 0: 7.20% | layer 1: 85.01%)

