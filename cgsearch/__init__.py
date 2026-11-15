"""Confidence-guided candidate search for AutoND (research code).

The AutoND cipher implementations, bias-score definition, evolutionary
optimizer and DBitNet model are imported unchanged from the pinned checkout in
``third_party/AutoND``. This package only adds:

* seeded (reproducible) data generation in AutoND's exact data format,
* a cipher-evaluation ledger for budget accounting,
* fixed-budget, adaptive (confidence-guided) and ablation scoring policies,
* a seeded DBitNet train/validate/test wrapper (needs TensorFlow 2.15),
* metrics and reproducibility manifests.
"""

__version__ = "0.1.0"
