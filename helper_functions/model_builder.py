"""
Contains PyTorch model code to instantiate a a series of different models.
"""
import torch
from torch import nn 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VanillaDNN(nn.Module):
  """Creates a Vanilla Deep Neural Network architecture.

  Args:
    input_shape: An integer indicating number of input channels.
    hidden_units: An integer indicating number of hidden units between layers.
    output_shape: An integer indicating number of output units.
  """
  def __init__(self, input_shape: int, hidden_units: int = 239, output_shape: int = 1) -> None:
      super().__init__()
      self.layer_stack = nn.Sequential(
       nn.Linear(in_features=input_shape, out_features=hidden_units),
       nn.ReLU(),
       nn.Linear(in_features=hidden_units, out_features=hidden_units),
       nn.ReLU(),
       nn.Linear(in_features=hidden_units, out_features=hidden_units),
       nn.ReLU(),
       nn.Linear(in_features=hidden_units, out_features=1)
      )

  def forward(self, x: torch.Tensor):
      # x = x.view(x.size(0), -1)
      output = self.layer_stack(x)
      return output
      # return self.classifier(self.conv_block_2(self.conv_block_1(x))) # <- leverage the benefits of operator fusion
  
class LSTM(nn.Module):
  def __init__(self, input_shape:int, hidden_units:int, num_stacked_layers: int):
    super().__init__()
    self.hidden_units = hidden_units
    self.num_stacked_layers = num_stacked_layers
    self.lstm = nn.LSTM(input_shape, hidden_units, num_stacked_layers, batch_first = True)
    self.fc = nn.Linear(hidden_units, 1) # fully connected layer to predict final closing value

  def forward(self, x):
    batch_size = x.size(0)
    #two values to initiate i.e. h0 and c0
    h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_units).to(device)
    c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_units).to(device)

    out, _ = self.lstm(x, (h0, c0))
    out = self.fc(out[:, -1, :])
    return out
