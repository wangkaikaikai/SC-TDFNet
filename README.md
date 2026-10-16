# SC-TDFNet

SC-TDFNet is a Subject-Conditioned Trend-Detail Fusion Network for interpretable continuous oxygen uptake (VO₂) prediction from non-gas cardiopulmonary exercise testing variables across clinical populations.

This repository provides the inference implementation of SC-TDFNet and a reference pretrained checkpoint.

## Model overview

SC-TDFNet combines subject-level static context, pathway-specific dynamic feature selection, multi-scale temporal modelling with causal convolutions, and reliability-gated trend-detail decoding. The model accepts a 32-step window with 17 input features and estimates VO₂ at the final window position. Intermediate trend, detail, feature-selection, and attention outputs are retained for model interpretation.

## Repository contents

- `src/sc_tdfnet/model.py`: SC-TDFNet architecture and checkpoint loader.
- `weights/sc_tdfnet_reference.pth`: reference pretrained checkpoint stored as a PyTorch state dictionary.
- `weights/SHA256SUMS`: SHA-256 checksum for verifying the checkpoint.
- `preprocessing.json`: input feature order, categorical encodings, and training-set normalisation parameters required by the checkpoint.
- `pyproject.toml`: package metadata and dependencies.

## Input and output

The required input tensor shape is `[batch, 32, 17]`. Features must follow the exact order listed in `preprocessing.json`. Continuous inputs are standardised with the provided training-set parameters. Categorical inputs use the specified integer encodings and are not standardised.

Continuous predictions are obtained by sliding the input window and retaining the prediction at each window endpoint.

Model outputs are in the standardised target space. Use the target mean and scale in `preprocessing.json` to restore predictions to mL·min⁻¹.

## Data

No participant-level or clinical data are included. Data access is subject to the data availability statement in the associated manuscript, institutional approvals, participant consent, and applicable privacy requirements.

## License

The source code and model checkpoint are released under the MIT License.

## Citation

Citation metadata will be updated when the article bibliographic record and DOI are available.
