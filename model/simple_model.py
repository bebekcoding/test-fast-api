from torch import nn

class MyModel(nn.Module):
    
    def __init__(self, input_size, output_size):
        super().__init__()

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

        self.input_layer = nn.Linear(input_size, 8)
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 8)
        self.output_layer = nn.Linear(8, output_size)

    def forward(self, X):
        h1 = self.relu(self.input_layer(X))
        h2 = self.relu(self.dropout(self.fc1(h1)))
        h3 = self.relu(self.fc2(h2))
        return self.output_layer(h3)
