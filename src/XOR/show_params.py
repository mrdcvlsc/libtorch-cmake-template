import torch
import pprint

class ActivatedLinearReLU(torch.nn.Module):
    def __init__(self):
        super(ActivatedLinearReLU, self).__init__()

        self.layer = torch.nn.Sequential(
            torch.nn.Linear(2, 2, True),
            torch.nn.BatchNorm1d(2),
            torch.nn.ReLU(True)
        )

    def forward(self, x):
        x = self.layer(x)
        return x

class ActivatedLinearSigmoid(torch.nn.Module):
    def __init__(self):
        super(ActivatedLinearSigmoid, self).__init__()

        self.layer = torch.nn.Linear(2, 1, True)
        self.activation = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.layer(x)
        x = self.activation(x)
        return x

class XorNet(torch.nn.Module):
    def __init__(self):
        super(XorNet, self).__init__()
    
        self.layer0 = torch.nn.Sequential(
            torch.nn.Linear(2, 2, True),
            torch.nn.Sigmoid(),
        )

        self.layer1 = ActivatedLinearReLU()
        self.layer2 = ActivatedLinearSigmoid()

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return x

if __name__ == "__main__":
    model = XorNet()

    pprint.pp(model.state_dict())

    # raw_state_dict = {}
    # for k, v in model.state_dict().items():
    #     print(f'name: {k}')
    #     # if isinstance(v, torch.Tensor):
    #     #     raw_state_dict[k] = (list(v.size()), v.numpy().tolist())
    #     #     break
