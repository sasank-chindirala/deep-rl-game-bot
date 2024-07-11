from torch import nn


class FC_Model_With_Coordinates(nn.Module):
    def __init__(self):
        super(FC_Model_With_Coordinates, self).__init__()
        self.input_layer = nn.Linear(in_features=4, out_features=100, bias=True)
        self.hidden_layer = nn.Linear(in_features=100, out_features=50, bias=True)
        self.hidden_layer1 = nn.Linear(in_features=50, out_features=25, bias=True)
        self.hidden_layer2 = nn.Linear(in_features=25, out_features=20, bias=True)
        self.hidden_layer3 = nn.Linear(in_features=20, out_features=15, bias=True)
        self.output_layer = nn.Linear(in_features=15, out_features=9, bias=True)
        self.activation_function = nn.ReLU()

    def forward(self, input_tensor):
        output = self.input_layer(input_tensor)
        output = self.activation_function(output)
        output = self.hidden_layer(output)
        output = self.activation_function(output)
        output = self.hidden_layer1(output)
        output = self.activation_function(output)
        output = self.hidden_layer2(output)
        output = self.activation_function(output)
        output = self.hidden_layer3(output)
        output = self.activation_function(output)
        output = self.output_layer(output)
        return output
