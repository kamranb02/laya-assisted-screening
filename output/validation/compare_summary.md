# Laya variant comparison (gold labels, cross-validated calibration)

| variant         | checkpoint      | question_style   |   max_len |   auc_w |   ece_raw |   ece_calibrated_cv |   precision_at_recall>=0.95 |   recall |   threshold_cal |   share_p_in_0.4_0.6 |   texts_trimmed_% |   seconds_per_paper | laya_version   |
|:----------------|:----------------|:-----------------|----------:|--------:|----------:|--------------------:|----------------------------:|---------:|----------------:|---------------------:|------------------:|--------------------:|:---------------|
| english_ab      | english         | choice_ab        |       512 |   0.897 |     0.258 |               0.022 |                       0.464 |    0.965 |            0.07 |                0.285 |            58.901 |               0.025 | 0.4.14         |
| english_noul    | english         | noul             |       512 |   0.589 |     0.168 |               0.014 |                       0.38  |    0.954 |            0.26 |                0.202 |            58.901 |               0.022 | 0.4.14         |
| multilingual_ab | multilingual    | choice_ab        |      1024 |   0.581 |     0.414 |               0.547 |                       0.386 |    0.859 |            0.01 |                0.118 |             7.068 |               0.02  | 0.4.14         |
| typed_ab        | typed-decisions | choice_ab        |      1024 |   0.897 |     0.242 |               0.044 |                       0.521 |    0.956 |            0.1  |                0.346 |             7.068 |               0.043 | 0.4.14         |

**Best by precision at recall >= 0.95: `typed_ab`** -> set in config.yaml: `screening.checkpoint: typed-decisions`, `screening.question_style: choice_ab`, `screening.max_len: 1024`.