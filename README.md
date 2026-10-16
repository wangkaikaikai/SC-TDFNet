# SC-TDFNet

SC-TDFNet is a Subject-Conditioned Trend-Detail Fusion Network for interpretable continuous oxygen uptake (VO2) prediction from non-gas cardiopulmonary exercise testing variables across clinical populations.

This repository is the public, manuscript-facing release of the inference-time model architecture and a reference checkpoint. It does not contain participant-level data, training logs, experiment histories, or internal model-selection utilities.

## Model overview

SC-TDFNet combines subject-level static context, pathway-specific dynamic feature selection, causal multi-scale temporal modelling, and reliability-gated trend-detail decoding. The model accepts a 32-step window with 17 input features and produces a continuous VO2 estimate at the final window position. Intermediate trend, detail, feature-selection, and attention outputs are retained for model interpretation.

## Repository contents

- `src/sc_tdfnet/model.py`: inference-time SC-TDFNet architecture and checkpoint loader.
- `weights/sc_tdfnet_reference.pth`: manuscript reference checkpoint stored as a PyTorch state dictionary.
- `weights/SHA256SUMS`: SHA-256 checksum for verifying the released checkpoint.
- `preprocessing.json`: input order, categorical encodings, and training-set normalisation parameters required by the checkpoint.
- `pyproject.toml`: minimal package metadata and dependency declaration.

## Input definition

The required tensor shape is `[batch, 32, 17]`. Features must follow the exact order listed in `preprocessing.json`. Continuous inputs are standardised with the released training-set parameters. Categorical inputs use the integer encodings listed in the same file and are not standardised.

The checkpoint output is in the standardised target space. The target mean and scale in `preprocessing.json` restore the prediction to mL min-1.

## Released checkpoint

The released file is the reference checkpoint associated with the manuscript. It is provided as a PyTorch state dictionary for research and verification purposes.

## Data

No participant-level or clinical data are included. Data access is subject to the final manuscript statement, institutional approvals, participant consent, and applicable privacy requirements.

## License

The source code is released under the MIT License. The checkpoint is provided for research use under the same repository license. This software is not a medical device and is not intended for diagnosis or clinical decision-making.

## Citation

Citation metadata will be updated when the article bibliographic record and DOI are available.
