import torch
import torch.nn as nn
import numpy as np
import pickle
import plotly.graph_objects as go
from torch.utils.data import TensorDataset, DataLoader
import process_data

NUM_AMINO_ACIDS = 20
PEPTIDE_LEN = 9

NUM_EPOCHS = 100
INNER_LAYER_NEW_SIZE = 30


def create_model_1st_dim():
    model = nn.Sequential(
        nn.Linear(PEPTIDE_LEN * NUM_AMINO_ACIDS, PEPTIDE_LEN * NUM_AMINO_ACIDS),
        nn.ReLU(),
        nn.Linear(PEPTIDE_LEN * NUM_AMINO_ACIDS, PEPTIDE_LEN * NUM_AMINO_ACIDS),
        nn.ReLU(),
        nn.Linear(PEPTIDE_LEN * NUM_AMINO_ACIDS, 2),  # 1 output layer
        nn.Softmax(dim=1)
    )
    return model


def create_model_2nd_dim():
    model = nn.Sequential(
        nn.Linear(PEPTIDE_LEN * NUM_AMINO_ACIDS, INNER_LAYER_NEW_SIZE),
        nn.ReLU(),
        nn.Linear(INNER_LAYER_NEW_SIZE, INNER_LAYER_NEW_SIZE),
        nn.ReLU(),
        nn.Linear(INNER_LAYER_NEW_SIZE, 2),  # 1 output layer
        nn.Softmax(dim=1)
    )
    return model


def create_and_train_model():
    model = create_model_2nd_dim()

    X, y = pickle.load(open("data/X_train.pkl", "rb")), pickle.load(open("data/Y_train.pkl", "rb"))
    # convert numpy arrays to torch tensors
    X = torch.tensor(X).float()
    y = torch.tensor(y).float()

    # convert one-hot encoded labels to class labels
    y_class_labels = np.argmax(y.numpy(), axis=1)

    # calculate class weights
    class_counts = np.bincount(y_class_labels)
    class_weights = torch.tensor([len(y_class_labels) / c for c in class_counts]).float()

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    # create a DataLoader for your data
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=1000, shuffle=True)  # adjust batch_size as needed

    # train and plot the loss throughout the training
    train_loss_values = []
    test_loss_values = []
    for n in range(NUM_EPOCHS):
        print(f"Epoch {n}")
        model.eval()
        test_loss_values.append(test_model(model, loss_fn))
        train_loss = 0
        model.train()
        for batch_X, batch_y in dataloader:  # iterate over batches
            y_pred = model(batch_X)
            loss = loss_fn(y_pred.view_as(batch_y), batch_y)
            train_loss = loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        train_loss_values.append(train_loss)

    torch.save(model, "my_model.pickle")
    return train_loss_values, test_loss_values, model


def test_model(model, loss_fn):
    # test the model and plot the test loss
    X, y = pickle.load(open("data/X_test.pkl", "rb")), pickle.load(open("data/Y_test.pkl", "rb"))
    X = torch.tensor(X).float()
    y = torch.tensor(y).float()
    with torch.no_grad():
        y_pred = model(X).squeeze()
        loss = loss_fn(y_pred, y)
        return loss.item()


def plot_loss(train_loss, test_loss):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=train_loss, mode='lines', name='Training Loss'))
    fig.add_trace(go.Scatter(y=test_loss, mode='lines', name='Testing Loss'))
    fig.update_layout(title='Training and Testing Loss', xaxis_title='Epochs', yaxis_title='Loss')
    fig.show()


def print_test_metrics(model):
    X, y = pickle.load(open("data/X_test.pkl", "rb")), pickle.load(open("data/Y_test.pkl", "rb"))
    X = torch.tensor(X).float()
    y = torch.tensor(y).float()
    with torch.no_grad():
        y_pred = model(X).squeeze()
        y_pred = torch.argmax(y_pred, dim=1)
        print("Predictions:", y_pred)
        y = torch.argmax(y, dim=1)
        print("Accuracy:", (y_pred == y).float().mean().item())
        print("Precision:", ((y_pred == 1) & (y == 1)).float().sum().item() / (y_pred == 1).float().sum().item())
        print("Recall:", ((y_pred == 1) & (y == 1)).float().sum().item() / (y == 1).float().sum().item())


def get_top_3_peptides(model):
    peptides, X = process_data.load_spike_data()
    # print the top 3 peptides with the highest probability of being a spike protein
    with torch.no_grad():
        y_pred = model(torch.tensor(X).float()).squeeze()
        y_pred = y_pred[:, 1].numpy()
        top_3_indices = np.argsort(y_pred)[::-1][:3]
        for i in top_3_indices:
            print(peptides[i], y_pred[i])


def main():
    # train_loss, test_loss, model = create_and_train_model()
    # plot_loss(train_loss, test_loss)
    # print_test_metrics(model)
    model = torch.load("my_model.pickle")
    get_top_3_peptides(model)


if __name__ == "__main__":
    main()
