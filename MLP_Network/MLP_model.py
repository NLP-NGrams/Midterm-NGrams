import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_layer_size=1,
                       hidden_layer_1_size=2,
                       hidden_layer_2_size=3,
                       hidden_layer_3_size=4,
                       output_layer_size=2):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_layer_size, hidden_layer_1_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_layer_1_size, hidden_layer_2_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_layer_2_size, hidden_layer_3_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_layer_3_size, output_layer_size)
        )

    def forward(self, x):
        logits = self.classifier(x)
        return logits

