# AirfRANS_competitions

This repo includes the Safran solutions to two machine learning competitions, based on the [AirfRANS](https://github.com/Extrality/AirfRANS) dataset.
The dataset is also available in [PLAID](https://plaid-lib.readthedocs.io/en/latest/) format [here](https://zenodo.org/records/12515084).

More details in the competition solution folders:
- ML4CFD_Neurisp2024
- ML4PHYSIM

# Environement

The solutions require the following dependencies:

- [LIPS](https://github.com/IRT-SystemX/LIPS)
- [muscat=2.0.2](https://muscat.readthedocs.io/en/latest/index.html)
- [airfrans](https://github.com/Extrality/AirfRANS)
- [triangle](https://rufat.be/triangle/)
- scikit-learn
- tqdm
- multiprocess
- pytorch

One can create a compatible environment using conda:

```bash
git clone https://github.com/IRT-SystemX/LIPS.git

conda create -n AirfRANScompetitions python=3.11
conda activate AirfRANScompetitions

cd /path/where/you/cloned/LIPS
pip install -e .

mamba install muscat=2.0.2 scikit-learn tqdm pytorch
pip install airfrans triangle multiprocess
```

# Solutions

Solutions can be computed by executing the "run.py" file in the ML4CFD_Neurisp2024 and ML4PHYSIM folders:

```bash
python run.py
```

Remark: these run.py files aim at giving tools for reproducing the results, but they have not been tested.
