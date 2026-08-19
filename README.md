# AMPSynth and ELK

This repository contains the main implementation of **AMPSynth**, a two-stage diffusion-based antimicrobial peptide (AMP) generation framework, together with **ELK**, an ESM-2--xLSTM--KAN model for AMP identification and minimum inhibitory concentration (MIC) prediction.

The current release includes the core code used for AMP generation, AMP classification, target-species MIC regression, and sequence-quality evaluation. Additional code associated with the manuscript is still being organized and will be uploaded in future updates.

**Note:** Some model weight files are too large to be uploaded at once and are currently being uploaded progressively. Please check back for future updates.

## Repository Structure

```text
.
├── data/
│   └── Training and evaluation datasets
│
├── Pretrain_Output/
│   └── Pretrained AMPSynth model weights
│
├── Finetune_Output/
│   └── Stage-2 fine-tuned AMPSynth model weights
│
├── ELK_Classifier_Output/
│   └── ELK classifier checkpoints and related outputs
│
├── MIC_Output_Results/
│   └── Unified four-species MIC regression checkpoints and outputs
│
├── single_MIC_Output_Results/
│   └── Checkpoints and outputs for species-specific MIC models
│
├── AMPSynth.py
├── ELK-CLASSIFIER.py
├── ELK-regression.py
├── MIC_EC.py
├── MIC_SA.py
├── MIC_PA.py
├── MIC_BS.py
├── kan.py
├── evaluate.py
└── requirements.txt
```

## Requirements

A CUDA-enabled GPU is recommended for model training and inference, especially for ESM-2-based feature extraction and ESMFold evaluation.

A recommended Python environment is provided in `requirements.txt`.

Install the required packages with:

```bash
pip install -r requirements.txt
```

## Important: Update File Paths Before Running

The current scripts contain local paths for training data, model checkpoints, generated sequences, and output directories.

**Before training or inference, please update all relevant file paths in the corresponding script to match your local environment.** Failure to update these paths may result in file-not-found errors or outputs being written to unintended locations.

The datasets required by the released scripts are provided in the `data/` directory.

---

## AMPSynth

`AMPSynth.py` contains the complete implementation of the AMPSynth generative framework and supports four major tasks:

1. Unconditional pretraining
2. Conditional fine-tuning
3. Unconditional AMP generation
4. Target-species-conditioned AMP generation

The execution mode is controlled by the following variable near the end of `AMPSynth.py`:

```python
RUN_MODE = "inference_unconditional"
```

Available modes are:

```python
RUN_MODE = "pretrain_uncond"          # Stage-1 unconditional pretraining
RUN_MODE = "finetune_condition"       # Stage-2 conditional fine-tuning
RUN_MODE = "inference_unconditional"  # Unconditional AMP generation
RUN_MODE = "inference_conditional"    # Target-species-conditioned generation
```

After selecting the desired mode and updating the corresponding input/output paths, run:

```bash
python AMPSynth.py
```

### Pretrained Weights

We provide model checkpoints for both stages:

- `Pretrain_Output/`: Stage-1 pretrained AMPSynth weights
- `Finetune_Output/`: Stage-2 conditionally fine-tuned AMPSynth weights

For unconditional inference, the script loads the Stage-1 pretrained model.

For conditional inference, the script loads the fine-tuned AMPSynth model from `Finetune_Output/` together with the pretrained sequence decoder from `Pretrain_Output/`.

---

## ELK: AMP Classification

`ELK-CLASSIFIER.py` implements the ELK classifier for binary AMP identification. The model combines ESM-2 sequence representations, xLSTM-based sequence modeling, and a KAN prediction head.

The script supports both training and prediction through:

```python
RUN_MODE = "train"
```

or

```python
RUN_MODE = "predict"
```

After updating the training-data, prediction-data, checkpoint, and output paths, run:

```bash
python ELK-CLASSIFIER.py
```

Pretrained classifier weights and related outputs are provided in:

```text
ELK_Classifier_Output/
```

---

## ELK: Target-Species MIC Regression

`ELK-regression.py` implements the ELK MIC regression model and supports MIC prediction for four bacterial species.

The script supports both training and prediction:

```python
RUN_MODE = "train"
```

or

```python
RUN_MODE = "predict"
```

For prediction, the target bacterial species is selected using:

```python
TARGET_BACTERIA = 3
```

The current ID mapping is:

| ID | Target species |
|---:|---|
| 0 | *Escherichia coli* (EC) |
| 1 | *Staphylococcus aureus* (SA) |
| 2 | *Pseudomonas aeruginosa* (PA) |
| 3 | *Bacillus subtilis* (BS) |

For example, to predict MIC values against *E. coli*:

```python
TARGET_BACTERIA = 0
```

After updating the required paths, run:

```bash
python ELK-regression.py
```

The unified four-species MIC regression weights and outputs are stored in:

```text
MIC_Output_Results/
```

### Species-Specific MIC Models

We additionally provide four scripts for independently trained species-specific MIC prediction models:

```text
MIC_EC.py
MIC_SA.py
MIC_PA.py
MIC_BS.py
```

These models use the same core ELK architecture but are trained independently for individual bacterial species.

Their model weights and output files are stored in:

```text
single_MIC_Output_Results/
```

As with the unified model, please update the corresponding data, checkpoint, prediction-input, and output paths before training or inference.

---

## KAN Implementation

`kan.py` contains the local implementation of the Kolmogorov--Arnold Network (KAN) layers used by ELK.

No separate external KAN package is required for this implementation.

---

## Evaluation of Generated Peptides

`evaluate.py` provides utilities for evaluating the quality of generated AMP sequences.

The current evaluation pipeline includes:

- Sequence similarity based on BLOSUM62 alignment
- ESM-2-based internal diversity
- Structural plausibility measured by ESMFold-predicted pLDDT
- ESM-2 pseudo-perplexity

Before running the evaluation script, update the paths of the generated sequences and reference/training sequences in `evaluate.py`.

Run:

```bash
python evaluate.py
```

ESMFold evaluation can require substantial GPU memory and may take considerably longer than the other sequence-level metrics.

---

## Data

The datasets required for the currently released training and evaluation scripts are provided in:

```text
data/
```

Please check the configuration section of each script for the expected file name, worksheet, sequence column, target label, or MIC field before running the code.

---

## Notes

- Several scripts currently use explicitly configured local paths. These must be changed before use on a different machine.
- GPU memory requirements depend on the selected ESM-2 model, batch size, and evaluation procedure.
- The provided checkpoints can be used directly for inference after the corresponding paths are configured.
- Additional code related to the manuscript is still being cleaned and organized and will be released in future updates.

## Citation

If you find this repository useful, please consider citing the corresponding work.

Citation information will be updated after publication.

## License

License information will be added in a future update.
