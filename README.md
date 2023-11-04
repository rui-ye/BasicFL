# Easy, Basic, and Clean Federated Learning Codebase

## How to run the example code

```
bash run_scripts/example.sh
```

## To run other experiments

- Go to `config.py` to check the arguments
- In `run_scripts/example.sh`, there are several common arguments in FL
- Currently supported algorithms: `FedAvg` / `FedProx` / `SCAFFOLD`.

## Others

If you find this repo useful, please feel free to cite us.

See other projects and papers at [Rui Ye's Homepage](https://rui-ye.github.io/).

Recommend the following papers of ours: (codes are not as clean as this repo)

- FedFM (TSP 2023) [\[Paper\]](https://arxiv.org/abs/2210.07615) / [\[Code\]](https://github.com/rui-ye/FedFM)

```
@article{ye2023fedfm,
  title={Fedfm: Anchor-based feature matching for data heterogeneity in federated learning},
  author={Ye, Rui and Ni, Zhenyang and Xu, Chenxin and Wang, Jianyu and Chen, Siheng and Eldar, Yonina C},
  journal={IEEE Transactions on Signal Processing},
  year={2023},
  publisher={IEEE}
}
```

- FedDisco (ICML 2023) [\[Paper\]](https://arxiv.org/abs/2305.19229) / [\[Code\]](https://github.com/MediaBrain-SJTU/FedDisco)

```
@article{ye2023feddisco,
  title={FedDisco: Federated Learning with Discrepancy-Aware Collaboration},
  author={Ye, Rui and Xu, Mingkai and Wang, Jianyu and Xu, Chenxin and Chen, Siheng and Wang, Yanfeng},
  journal={arXiv preprint arXiv:2305.19229},
  year={2023}
}
```

- pFedGraph (ICML 2023) [\[Paper\]](https://openreview.net/forum?id=33fj5Ph3ot) / [\[Code\]](https://github.com/MediaBrain-SJTU/pFedGraph)

```
@article{ye2023personalized,
  title={Personalized Federated Learning with Inferred Collaboration Graphs},
  author={Ye, Rui and Ni, Zhenyang and Wu, Fangzhao and Chen, Siheng and Wang, Yanfeng},
  year={2023}
}
```