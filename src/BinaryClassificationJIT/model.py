import os
import torch

class MultiClassClassification(torch.nn.Module):
    def __init__(self):
        super(MultiClassClassification, self).__init__()

        # Input: Bx1x28x28

        self.conv_relu_stack = torch.nn.Sequential(
            # feature extraction block without downsampling
            torch.nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True),
            torch.nn.BatchNorm2d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            torch.nn.ReLU(inplace=True),
            # Bx4x28x28

            # feature processing block
            torch.nn.Conv2d(in_channels=4, out_channels=4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True),
            torch.nn.BatchNorm2d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            torch.nn.ReLU(inplace=True),
            # Bx4x28x28

            # feature downsampling using max pooling
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0), dilation=1, ceil_mode=False),
            # Bx4x14x14
        )

        self.flatten = torch.nn.Flatten()

        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(4*14*14, 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(128, 10),
            torch.nn.Softmax(dim=1),
        )


    def forward(self, x):

        x = self.conv_relu_stack(x)
        x = self.flatten(x)
        x = self.linear_relu_stack(x)

        return x

if __name__ == "__main__":

    MODEL_SAVE_PATH = os.path.join('checkpoint_v_mcc_model.pth')

    if os.path.isfile(MODEL_SAVE_PATH):
        print('Model Already Existing')
    else:
        model = MultiClassClassification()
        model.to("cpu")
        model.eval()

        # FOR MORE INFO WHETHER TO USE TRACING OR SCRIPTING SEE LINK:
        # https://pytorch.org/tutorials/advanced/cpp_export.html

        traced_script_module = torch.jit.trace(model, torch.rand(1, 1, 28, 28))
        torch.jit.save(traced_script_module, MODEL_SAVE_PATH)
        
        print('Jit\'ed Model Created Existing')
