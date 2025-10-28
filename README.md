# Task Inclusion Estimation

![alt text](images/data_model)

This repository contains all the code on experiments regarding the paper [Statistical deficiency for task inclusion estimation](https://arxiv.org/abs/2503.05491) accepted at ACL 2025 (Vienna Austria) in the main track.

## 1. Installation

Get uv for fast package installation.

First, you need to install `uv`

```bash
CURL=$(which curl)
${CURL} -LsSf https://astral.sh/uv/install.sh | sh
```

Once uv is installed you can `sync`, which will create a virtual environnement.

```bash
uv sync
```

## 2. Getting data

Data will be stored into a `data` folder. If you have a bash env 

```bash
mkdir data
cd data
git clone https://gitlab.lis-lab.fr/talep-public/acl2025
cd ..
```

## 3. Launching scripts

In the [`scripts`](scripts) folder, there is multiple examples of how to use this repository.

```bash
bash ./scripts/launch_finetuning.sh
```

```bash
bash ./scripts/launch_dump_embeddings.sh
```

```bash
bash ./scripts/launch_information_sufficiency.sh
```

## How to cite ?

ACL - Anthology
```bib
@inproceedings{fosse-etal-2025-statistical,
    title = "Statistical Deficiency for Task Inclusion Estimation",
    author = {Fosse, Lo{\"i}c  and
      Bechet, Frederic  and
      Favre, Benoit  and
      Damnati, G{\'e}raldine  and
      Lecorv{\'e}, Gw{\'e}nol{\'e}  and
      Darrin, Maxime  and
      Formont, Philippe  and
      Piantanida, Pablo},
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-long.18/",
    doi = "10.18653/v1/2025.acl-long.18",
    pages = "382--415",
    ISBN = "979-8-89176-251-0",
}
```

Arxiv
```bib
@article{fosse2025statistical,
  title={Statistical Deficiency for Task Inclusion Estimation},
  author={Fosse, Lo{\"\i}c and B{\'e}chet, Fr{\'e}d{\'e}ric and Favre, Beno{\^\i}t and Damnati, G{\'e}raldine and Lecorv{\'e}, Gw{\'e}nol{\'e} and Darrin, Maxime and Formont, Philippe and Piantanida, Pablo},
  journal={arXiv preprint arXiv:2503.05491},
  year={2025}
}
```