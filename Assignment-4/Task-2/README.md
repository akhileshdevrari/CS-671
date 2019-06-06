## Variable sized Input sequence
We padded each input seq to length of 10.

	Input - [0.001, 0.002, 0.003, 0.004, 0.005]
	Padded Input - [0., 0., 0., 0., 0., 0.001, 0.002, 0.003, 0.004, 0.005]

##  Architecture

	Task 1:
		SimpleRNN was used for this task.
		We tried to generalize model by training it on both sin & triangle wave data.

	Task 2:
		Bidirectional LSTM was used for AP problem.
		We trained our model on AP progressions with common differernce 1,2,3,4,5.

## Results

All graphs are in results folder.
