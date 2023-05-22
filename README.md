## Puffin Manuscript

This repository contains the code and data required for reproducing the analyses in the Puffin manuscript. For using Puffin and training Puffin models, please visit our main [repository](https://github.com/jzhoulab/puffin). 
For most use cases, we highly recommend running Puffin from our webserver, [tss.zhoulab.io](https://tss.zhoulab.io) or [puffin.zhoulab.io](https://puffin.zhoulab.io).

Most of the analyses code are provided in jupyter notebook format. Each jupyter notebook contains a series of analyses and typically generates multiple plots for related analyses.

The jupyter notebooks are grouped by topics:

- [Puffin_analysis.ipynb](Puffin_analysis.ipynb) : The main analysis notebook for most Puffin-based analyses.
- [Motif_contribution.ipynb](Motif_contribution.ipynb) : Motif contribution score-based analyses including analysis of insertion screen results.
- [Puffin_insilicoKO.ipynb](Puffin_insilicoKO.ipynb) : In silico KO analyses.
- [NFY_TATA_experiment_insilico](NFY_TATA_experiment_insilico) : Motif insertion and deletion effect analysis.
- [Trinucleotide_effects.ipynb](Trinucleotide_effects.ipynb) : Normalizing and analyzing trinucleotide effects (raw trinucleotide effect scores have to be properly normalized before any analysis, as detailed in Supplementary Note 2 of the manuscript).
- [Evolution_conservation.ipynb](Evolution_conservation.ipynb) : Evolutionary conservation analysis.
- [Evolution_mouse_comparison.ipynb](Evolution_mouse_comparison.ipynb) : Human and mouse sequence dependency comparison analysis. 
- [Puffin-D_analysis.ipynb](Puffin-D_analysis.ipynb) : Analysis of Puffin-D synthetic promoter screen results.
- [train/FANTOM_CAGE_biascorrection.ipynb](./train/FANTOM_CAGE_biascorrection.ipynb) : Preprocessing of FANTOM CAGE data.
- [train/train_puffin_D.py](./train/train_puffin_D.py) : Training code for Puffin-D.
- [train/train_puffin_stage_1.py](./train/train_puffin_stage_1.py) : Training code for Puffin.
- [train/train_puffin_stage_2.py](./train/train_puffin_stage_2.py) : Training code for Puffin.
- [train/train_puffin_stage_3.py](./train/train_puffin_stage_3.py) : Training code for Puffin.
- [train/Process_models.ipynb](./train/Process_models.ipynb) : Processing of models during multi-stage training.
- [train/train_puffin_mouse.py](./train/train_puffin_mouse_stage_1.py) : Training code for mouse Puffin for human-mouse comparison.
- [puffin_D_synthetic_screen.py](puffin_D_synthetic_screen.py) : Code for the Puffin-D synthetic promoter screen.
- [puffin_D_insertion_screen.py](puffin_D_insertion_screen.py) : Code for the Puffin-D promoter insertion screen.



### Dependencies
Other than [Puffin dependencies](https://github.com/jzhoulab/puffin#installation), you will also need jupyter, rpy2 python packages which can be installed with Anaconda or pip. For R packages, we will use data.table, ggplot2, patchwork, ggridges, ggrastr, ggthemes, and ggExtra.

### Data
You will need resource files for reproducing the analyses, and we have provided these files through Zenodo and can be downloaded using the commands below
```
#under the puffin_mansucript directory
wget https://zenodo.org/record/7954971/files/resources.tar.xz
tar xf ./resources.tar.xz
```

# puffin_manuscript
# puffin_manuscript
# puffin_manuscript
# puffin_manuscript
