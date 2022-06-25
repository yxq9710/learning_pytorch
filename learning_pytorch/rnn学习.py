from torch import nn

rnn = nn.RNN(input_size=100, hidden_size=20, num_layers=3)
print(rnn)
