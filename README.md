A recurrent neural network model with key-value episodic memory buffer watching _This is Us_ Season 1 and performing next scene prediction task. Model's representations and memory retrieval are compared with the human's event-by-event causal relationship ratings and memory retrieval. 

- clip: CLIP embedding time series of _i)_ scenes in episodes 2 to 18 of _This is Us_ Season 1, and _ii)_ scenes in segmented events 1 to 48 of episode 1, _This is Us_ Season 1.
- code:
- data:
- input: 
- model: implements EM-GRU, EM-GRU with shuffled memory, EM-GRU with fixed input-to-key and input-to-query mappings, and GRU without the EM buffer
