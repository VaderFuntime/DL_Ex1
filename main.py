import torch
import torch.nn as nn
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

NUM_AMINO_ACIDS = 20
PEPTIDE_LEN = 9

NUM_EPOCHS = 100


def create_and_train_model():
    model = nn.Sequential(
        nn.Linear(PEPTIDE_LEN * NUM_AMINO_ACIDS, PEPTIDE_LEN * NUM_AMINO_ACIDS),
        nn.ReLU(),
        nn.Linear(PEPTIDE_LEN * NUM_AMINO_ACIDS, PEPTIDE_LEN * NUM_AMINO_ACIDS),
        nn.ReLU(),
        nn.Linear(PEPTIDE_LEN * NUM_AMINO_ACIDS, 1),  # 1 output layer
        nn.Sigmoid()
    )

    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    X, y = pickle.load(open("data/X_train.pkl", "rb")), pickle.load(open("data/Y_train.pkl", "rb"))
    # convert numpy arrays to torch tensors
    X = torch.tensor(X).float()
    y = torch.tensor(y).float()

    # train and plot the loss throughout the training
    loss_values = []
    for n in range(NUM_EPOCHS):
        print(f"Epoch {n}")
        y_pred = model(X).squeeze()
        loss = loss_fn(y_pred, y)
        loss_values.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.save(model, "my_model.pickle")
    return loss_values


def test_model():
    # test the model and plot the test loss
    model = torch.load("my_model.pickle")
    X, y = pickle.load(open("data/X_test.pkl", "rb")), pickle.load(open("data/Y_test.pkl", "rb"))
    X = torch.tensor(X).float()
    y = torch.tensor(y).float()
    y_pred = model(X).squeeze()
    loss_fn = nn.BCELoss()
    loss = loss_fn(y_pred, y)
    return loss.item()

def plot_loss(train_loss, test_loss):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=train_loss, mode='lines', name='Training Loss'))
    fig.add_trace(go.Scatter(y=[test_loss]*len(train_loss), mode='lines', name='Testing Loss'))
    fig.update_layout(title='Training and Testing Loss', xaxis_title='Epochs', yaxis_title='Loss')
    fig.show()




if __name__ == "__main__":
    train_loss = create_and_train_model()
    test_loss = test_model()
    plot_loss(train_loss, test_loss)
