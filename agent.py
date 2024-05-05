import mlx.core as mx
import mlx.nn as nn


class MarioNet(nn.Module):
    """Mini CNN structure adapted from the original PyTorch DQN tutorial.
    input -> (conv2d + relu) x 3 -> flatten -> (dense + relu) x 2 -> output
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        c, h, w = input_dim

        if h != 84:
            raise ValueError(f"Expecting input height: 84, got: {h}")
        if w != 84:
            raise ValueError(f"Expecting input width: 84, got: {w}")

        self.online_conv = self.__build_cnn_conv(c)
        self.online_fc = self.__build_cnn_fc(output_dim)
        self.target_conv = self.__build_cnn_conv(c)
        self.target_fc = self.__build_cnn_fc(output_dim)

        # Initialize Q_target parameters to be the same as Q_online
        self.target_conv.update(self.online_conv.parameters())
        self.target_fc.update(self.online_fc.parameters())

        # Q_target parameters are frozen.
        self.target_conv.freeze()
        self.target_fc.freeze()

    def __call__(self, input, model):

        # Reshape input to (batch, channels, height, width)
        input = input.transpose(0, 2, 3, 1)

        if model == "online":
            output = self.online_conv(input)
            output = output.flatten(start_axis=1)
            output = self.online_fc(output)
            return output

        elif model == "target":
            output = self.target_conv(input)
            output = output.flatten(start_axis=1)
            output = self.target_fc(output)
            return output

    def __build_cnn_conv(self, c):
        return nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

    # Workaround for the lack of nn.Flatten in MLX
    def __build_cnn_fc(self, output_dim):
        return nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )
