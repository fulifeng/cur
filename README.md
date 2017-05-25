The experimental codes for SIGIR'17 paper "Computational Social Indicators: A Case Study of Chinese University Ranking", including the implementation of the proposed Graph-based Multi-channel Ranking (GMR) model and several baselines.

## Data

All the data and calculated Laplacian matrices mentioned in the paper are under the data folder. For detailed data collection and Laplacian computation, please refer to the paper.

## Code

| Method | Script |
| :-----------: | :-----------: |
| Graph-based Multi-channel Ranking (GMR) | run_gmr.py |
| Heuristic Ranking | run_heuristic.py |
| Early Fusion | run_early_fusion.py |
| Late Fusion | run_late_fusion.py |
| Joint Learning | run_joint_learning.py |
| Subspace Learning | run_subspace_learning.py |

If you want to use any of the aforementioned methods, just run the corresponding script with:

        `python xxx.py`

It should be noted that you need to edit the `basic_dir` variable in each script to be the absolute path where the data folder is cloned. Currently, all the methods are initialized with optimal hyperparameters tunned with grid search as described in the paper. If you want to further tune the hyperparameters, set the `tune` variable in the corresponding function to be `True` first.  

## Cite

If you use the code, please kindly cite the following paper:
@inproceedings{fuli2017computational,
  title={Computational social indicators: a case study of Chinese university ranking},
  author={Feng, Fuli and Nie, Liqiang and Wang, Xiang, and Hong, Richang and Chua Tat-Seng},
  booktitle={Proceedings of International ACM SIGIR Conference on Research and Development in Informaion Retrieval},
  year={2017},
  organization={ACM}
}

## Contact

fulifeng93@gmail.com
