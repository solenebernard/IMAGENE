# exp

Data of the experiment, read and written by the scripts of `../code/`.
With the default `code/config.yaml`, this folder is the experiment (`rootpath: '../'`, `exp_path: 'exp/'`).

```
exp/
├── Live/                 brightfield images, masks and tracks of each condition (not included: too large)
├── Fish/                 gene expression tables of step 2: {CONDITION}.csv (the smFISH images are not included)
├── alignment/            matched live and fixed cells of step 3: {CONDITION}-alignment.csv
└── live_cell_features/   feature tables of steps 1.4 and 4, for each condition
```

The tables of the two conditions activated for 24 hours (`3-24H`, `4-24H`) are included. They are the inputs of step 5.1:
`python3 -m 5_POC_activation.1_concatenate_features`, run from `code/`, rebuilds `code/6_results/dataset.csv` from them.

See "Data layout" in `../readme.md` for the complete layout of an experiment.
