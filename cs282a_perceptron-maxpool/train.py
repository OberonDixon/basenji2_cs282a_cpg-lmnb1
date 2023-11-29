from ds_util import get_dataset
from model import MLPModel
import numpy as np
import torch
import h5py
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

f, dset = get_dataset()

train_inds = np.zeros(len(dset), dtype=bool)
val_inds = np.zeros(len(dset), dtype=bool)
test_inds = np.zeros(len(dset), dtype=bool)
values = np.unique(dset)

for value in values:
    value_inds = np.nonzero(dset == value)[0]
    np.random.shuffle(value_inds)
    n = int(0.8 * len(value_inds))
    val_split = int(0.9 * len(value_inds))

    train_inds[value_inds[:n]] = True
    val_inds[value_inds[n:val_split]] = True
    test_inds[value_inds[val_split:]] = True

labels_full = h5py.File('dataset_14-lmnb1_4-cpg.h5')['single_bin']

train_tensor_dset = torch.utils.data.TensorDataset(torch.Tensor(dset[train_inds]), torch.Tensor(labels_full[train_inds]))
val_tensor_dset = torch.utils.data.TensorDataset(torch.Tensor(dset[val_inds]), torch.Tensor(labels_full[val_inds]))
test_tensor_dset = torch.utils.data.TensorDataset(torch.Tensor(dset[test_inds]), torch.Tensor(labels_full[test_inds]))

training_loader = torch.utils.data.DataLoader(train_tensor_dset, batch_size=4, shuffle=True)
validation_loader = torch.utils.data.DataLoader(val_tensor_dset, batch_size=4, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_tensor_dset, batch_size=4, shuffle=False)

curr_model = MLPModel()
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(curr_model.parameters(), lr=0.002, momentum=0.8)


def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.
    for i, data in enumerate(training_loader):
        inputs, labels = data

        optimizer.zero_grad()
        outputs = curr_model(inputs)

        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if i % 1000 == 999:
            last_loss = running_loss / 1000
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss


timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/mlp_model_{}'.format(timestamp))

epoch_number = 0
EPOCHS = 15
best_vloss = 1_000_000.

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    curr_model.train(True)
    avg_loss = train_one_epoch(epoch_number, writer)
    running_vloss = 0.0
    curr_model.eval()

    with torch.no_grad():
        for i, vdata in enumerate(validation_loader):
            vinputs, vlabels = vdata
            voutputs = curr_model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch_number + 1)
    writer.flush()

    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'model_{}_{}'.format(timestamp, epoch_number)
        torch.save(curr_model.state_dict(), model_path)

    epoch_number += 1