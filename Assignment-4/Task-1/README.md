## Variable sized Input sequence
We padded each input seq to length of 10.

	Input - [0.001, 0.002, 0.003, 0.004, 0.005]
	Padded Input - [0., 0., 0., 0., 0., 0.001, 0.002, 0.003, 0.004, 0.005]

##  Architecture

	SimpleRNN was used for this task.
	Bidirectional LSTM was used for AP problem.
	For Collatz, Bidirectional LSTM was used.

## Results

All graphs are in results folder.
