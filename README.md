# NSL-Spec
This is NSL-Spec rep.

# Speculative Sampling

An implementation of speculative sampling as described in the paper [SpecInfer: Accelerating Large Language Model Serving with Tree-based Speculative Inference and Verification](https://arxiv.org/abs/2305.09781).
We will perform performance optimization based on it.


## Requirements

```bash
transformers==4.37.0
torch
accelerate

# if you use llama 
# sentencepiece
# protobuf
```

## Catalog Contents

- /
  - `LICENSE`
  - `README.md`
  - `batch_main.py`
  - `config.json` - configuration of inference
  - `log/`
    - ... (some log files)
  - `models/`
    - ... (support models)
  - `one_draft_spec.py` - one draft spec test file
  - `outputs/`
    - ... (performance test results on the dataset)
  - `requirements.txt`
  - `scTest/`
    - ... (some test scripts)
  - `timeline/`
    - ... (timeline utils)
  - `tree_main.py`
  - `utils/`
    - ... (all the toolkit you need for speculative reasoning)


### How to use

```bash
python batch_main.py 
python tree_main.py 
```
Note that when changing the model, you need to modify the model loading class in batch_main.py and tree_main.py.



**Target Model - `facebook/opt-13b`**  
**Draft Model - `facebook/opt-1.3b`**

| Config            | Speedup (Set 1) | Speedup (Set 2) | Average Speedup |
|-------------------|-----------------|-----------------|-----------------|
| Temperature = 0   |             |             |             |
| Temperature = 0.5 |             |             |             |

**Target Model - `facebook/opt-6.7b`**  
**Draft Model - `facebook/opt-1.3b`**

| Config            | Speedup (Set 1) | Speedup (Set 2) | Average Speedup |
|-------------------|-----------------|-----------------|-----------------|
| Temperature = 0   |             |             |             |
| Temperature = 0.5 |             |             |             |

