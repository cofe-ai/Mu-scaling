from torch.nn import Linear

class MuReadout(Linear):
    '''Drop-in replacement for all output linear layers.

    An "output" linear layer is one that maps from a width dimension (e.g.,
    `d_model` in a Transformer) to a non-width dimension (e.g., vocab size).

    This layer implements the version of μP with a 1/width multiplier and a
    constant variance initialization for both weights and biases.
    '''
    def __init__(self, *args, readout_zero_init=False, output_mult=1.0, width_mult=1.0, **kwargs):
        self.output_mult = output_mult
        self.readout_zero_init = readout_zero_init
        self.width_mult_val = width_mult
        super().__init__(*args, **kwargs)
    
    def width_mult(self):
        return self.width_mult_val

    def reset_parameters(self) -> None:
        if self.readout_zero_init:
            self.weight.data[:] = 0
            if self.bias is not None:
                self.bias.data[:] = 0
        else:
            super().reset_parameters()

    def forward(self, x):
        return super().forward(
            self.output_mult * x / self.width_mult())


if __name__ == '__main__':
    pass
