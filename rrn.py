import torch
import torch.nn as nn

torch.manual_seed(1)

# Create the RNN layer
rnn_layer = nn.RNN(input_size=5, hidden_size=2, num_layers=1, batch_first=True)

# Extract weights and biases
w_xh = rnn_layer.weight_ih_l0  # shape: (2, 5)
w_hh = rnn_layer.weight_hh_l0  # shape: (2, 2)
b_xh = rnn_layer.bias_ih_l0    # shape: (2,)
b_hh = rnn_layer.bias_hh_l0    # shape: (2,)

print('W_xh shape: ', w_xh.shape)
print('W_hh shape: ', w_hh.shape)
print('b_xh shape: ', b_xh.shape)
print('b_hh shape: ', b_hh.shape)

# Input sequence of 3 timesteps, each with 5 features
x_seq = torch.tensor([[1.0]*5, [2.0]*5, [3.0]*5]).float()
print(f'x_seq: {x_seq}')

# Use the RNN layer normally
output, hn = rnn_layer(torch.reshape(x_seq, (1, 3, 5)))
print('output ', output)
print('hn ', hn)

# Manual forward computation
out_man = []

for t in range(3):
    xt = torch.reshape(x_seq[t], (1, 5))  # reshape to (1, 5) to match matmul shape

    print(f'Time step {t} =>')
    print('  Input   :', xt.numpy())

    # Compute input * input-hidden weights + input bias
    # xt shape: (1, 5), w_xh.T shape: (5, 2) → result: (1, 2)
    ht = torch.matmul(xt, torch.transpose(w_xh, 0, 1)) + b_xh
    print('      Hidden         :', ht.detach().numpy())

    if t > 0:
        prev_h = out_man[t-1]  # previous timestep hidden state
    else:
        prev_h = torch.zeros((ht.shape))  # zero init for t = 0

    # Add the recurrent (hidden-to-hidden) contribution:
    # prev_h shape: (1, 2), w_hh.T shape: (2, 2) → result: (1, 2)
    # Then add hidden bias
    ot = ht + torch.matmul(prev_h, torch.transpose(w_hh, 0, 1)) + b_hh

    # Apply non-linearity (tanh)
    ot = torch.tanh(ot)

    # Store this output to feed into next step
    out_man.append(ot)

    # Print manual vs actual RNN output
    print('   Output (manual) :', ot.detach().numpy())
    print('   RNN output      :', output[:, t].detach().numpy())
    print()
