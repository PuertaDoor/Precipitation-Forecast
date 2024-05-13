import torch
import torch.nn as nn

__all__ = ['ConvGRU']

"""
Modified from https://github.com/ndrplz/ConvLSTM_pytorch/blob/master/convlstm.py
"""


class ConvGRUCell(nn.Module):
    def __init__(self, input_data, input_dim, hidden_dim, kernel_size, bias, for_metnet):
        super(ConvGRUCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        # Dynamic spatial dimension adjustments
        if input_data == 'gdaps_kim':
            h, w = 50, 65
        elif input_data == 'gdaps_um':
            h, w = 151, 130
        if for_metnet:
            h, w = h // 2, w // 2
        
        self.height, self.width = h, w

        # Gates convolution: Combines reset and update gates
        self.conv_gates = nn.Conv2d(in_channels=input_dim + hidden_dim,
                                    out_channels=2 * hidden_dim,
                                    kernel_size=self.kernel_size,
                                    padding=self.padding,
                                    bias=self.bias)

        # Candidate state convolution
        self.conv_candidate = nn.Conv2d(in_channels=input_dim + hidden_dim,
                                         out_channels=hidden_dim,
                                         kernel_size=self.kernel_size,
                                         padding=self.padding,
                                         bias=self.bias)

        self.bn_gates = nn.BatchNorm2d(2 * hidden_dim, track_running_stats=False)

    def forward(self, input_tensor, h_cur):
        h_cur = torch.zeros(input_tensor.size(0), self.hidden_dim, input_tensor.size(2), input_tensor.size(3),
                            device=input_tensor.device)

        combined = torch.cat([input_tensor, h_cur], dim=1)
        gates = self.bn_gates(self.conv_gates(combined))
        r_gate, z_gate = torch.split(gates, self.hidden_dim, dim=1)
        r_gate = torch.sigmoid(r_gate)
        z_gate = torch.sigmoid(z_gate)

        combined_reset = torch.cat([input_tensor, r_gate * h_cur], dim=1)
        cc_candidate = torch.tanh(self.conv_candidate(combined_reset))

        h_next = z_gate * h_cur + (1 - z_gate) * cc_candidate
        return h_next
    
    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv_gates.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv_gates.weight.device))

class ConvGRU(nn.Module):
    def __init__(self, input_data, window_size, input_dim, hidden_dim, kernel_size, num_layers, num_classes, batch_first=False, bias=True, return_all_layers=False, for_metnet=False):
        super(ConvGRU, self).__init__()

        self.num_layers = num_layers
        self.batch_first = batch_first
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dim
            cell_list.append(ConvGRUCell(input_data=input_data,
                                         input_dim=cur_input_dim,
                                         hidden_dim=hidden_dim,
                                         kernel_size=kernel_size,
                                         bias=bias,
                                         for_metnet=for_metnet))

        self.cell_list = nn.ModuleList(cell_list)
        self.out_conv = nn.Conv2d(hidden_dim, num_classes, kernel_size=1)

    def forward(self, input_tensor, hidden_state=None):
        if not self.batch_first:
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))

        last_state_list = []
        current_input = input_tensor

        for layer_idx, cell in enumerate(self.cell_list):
            h = hidden_state[layer_idx]
            output_inner = []
            for time_step in range(input_tensor.size(1)):
                h = cell(input_tensor=current_input[:, time_step, :, :, :], h_cur=h)
                output_inner.append(h)
            current_input = torch.stack(output_inner, dim=1)
            last_state_list.append(h)

        last_layer_output = last_state_list[-1]
        out = self.out_conv(last_layer_output)

        return out

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states