A recurrent neural network with key-value episodic memory buffer watching [_This is Us_](https://en.wikipedia.org/wiki/This_Is_Us) Season 1 and performing next scene prediction task. Model parameter representations are compared with the human-rated causal relationships between events, to test whether the model represents events based on causal structure. Memories retrieved by the model are compared with memories retrieved by human participants to test whether the model retrieves meaningful past events.

- **clip**: CLIP embedding time series of _i)_ scenes in episodes 2 to 18 of _This is Us_ Season 1, and _ii)_ scenes in segmented events 1 to 48 of episode 1, _This is Us_ Season 1.
- **code**: modelfit*.py trains the model. code_*.py runs analyses and generates figures in the manuscript.
- **data**: includes _i)_ event orders for the three scrambled-order groups, _ii)_ event-by-event causal relationship ratings in the events' original event order, _iii)_ event-by-event human memory retrieval matrix in the events' original event order. the data were published in [Song et al. (2025)](https://www.biorxiv.org/content/10.1101/2025.03.12.642853v1) and also deposited in the repository [/memoryaha/data/](https://github.com/hyssong/memoryaha/tree/main/data).
- **input**: input used for model train & test. PCA was applied to the CLIP embeddings to reduce the dimension to 50, and the data were summarized to be used for model train & test.
- **model**: emKeyValue.py implements EM-GRU and EM-GRU with shuffled memory. gru.py implements GRU without the EM buffer.

## installation
To run the models and codes, it is necessary to install the Python packages included in environment.yml. We recommend installing conda and executing the following commands. This takes less than a minute in a standard laptop.
```bash
conda env create -f environment.yml
conda activate emgru

## model training
The following command trains the EM-GRU model on episodes 2-18, for 50 iterations, using a selected random seed. This takes approximately 2 hours on a standard laptop without GPU.
```bash
python code/modelfit.py original 1 0.5 0.1
# python [condition] [seed number] [alpha] [tau]

