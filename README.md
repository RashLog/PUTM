# PUTM
Official PyTorch Implementation of PUTM: Prototypes-oriented Transductive Few-shot Learning with Conditional Transport(ICCV 2023)

## Download Features
...

## Code Structure
```sh
├── features
│   ├── resnet18
│   │   ├── cub
│   │   └── mini
│   ├── wideres
│   │   └── mini
│   └── wrn_s2m2
│       ├── cifar
│       ├── cub
│       ├── mini
└── PUTM-main
    ├── cache
    ├── checkpoints
    ├── config
    ├── datasets
    ├── methods
    ├── models
    └── split
    ...
```



## Evaluation

Firstly, you should modify the configuration file **"config/base_config.yaml"** for evaluation on different settings. (dataset, balanced/imbalanced, backbone, etc)

```sh
cd PUTM-main 

python eval.py --base_config config/base_config.yaml --method_config config/[balanced, dirichlet]/methods_config/[method_name].yaml
```

For example, if you want to evaluate PUTM on imbalanced setting, then use the following config file, 

```yaml
MODEL:
  arch: 'wrn_s2m2' # ('resnet18', 'wideres', 'wrn_s2m2')

DATA:
  dataset: 'mini'
  batch_size_loader: 256
  enlarge: True
  num_workers: 4
  disable_random_resize: False
  jitter: False
  path: 'data'

EVAL:
  evaluate: True      # Set to True to evaluate methods
  number_tasks: 1000 # Number of tasks to evaluate
  batch_size: 1000
  n_ways: 5
  n_query: 15 # Balanced case: 15 query data per class
  balanced: 'dirichlet' # ('balanced' | 'dirichlet')
  alpha_dirichlet: 2
  model_tag: 'best'
  plt_metrics: ['accs']
  shots: [1, 3, 5]
  used_set: 'test'
  fresh_start: False

```

and run this command, 

```sh
python eval.py --base_config config/base_config.yaml --method_config   config/dirichlet/methods_config/putm.yaml
```

