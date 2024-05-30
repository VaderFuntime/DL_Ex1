import torch
import torch.nn as nn
import numpy as np
import pickle
import plotly.graph_objects as go
from torch.utils.data import TensorDataset, DataLoader
import process_data

NUM_AMINO_ACIDS = 20
PEPTIDE_LEN = 9

NUM_EPOCHS = 30
INNER_LAYER_NEW_SIZE = 30

BATCH_SIZE = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_model_1st_dim():
    """
    Create a model with 2 hidden layers, with dimensions equal to the input
    :return:
    """
    model = nn.Sequential(
        nn.Linear(PEPTIDE_LEN * NUM_AMINO_ACIDS, PEPTIDE_LEN * NUM_AMINO_ACIDS),
        nn.ReLU(),
        nn.Linear(PEPTIDE_LEN * NUM_AMINO_ACIDS, PEPTIDE_LEN * NUM_AMINO_ACIDS),
        nn.ReLU(),
        nn.Linear(PEPTIDE_LEN * NUM_AMINO_ACIDS, 2),
    )
    return model


def create_model_2nd_dim():
    """
    Create a model with 2 hidden layers, each with smaller inner layer size
    :return:
    """
    model = nn.Sequential(
        nn.Linear(PEPTIDE_LEN * NUM_AMINO_ACIDS, INNER_LAYER_NEW_SIZE),
        nn.ReLU(),
        nn.Linear(INNER_LAYER_NEW_SIZE, INNER_LAYER_NEW_SIZE),
        nn.ReLU(),
        nn.Linear(INNER_LAYER_NEW_SIZE, 2),
    )
    return model


def load_data(file_path):
    """
    Load the data from the file
    :param file_path:
    :return:
    """
    data = pickle.load(open(file_path, "rb"))
    return torch.tensor(data).float()


def create_and_train_model():
    """
    Create and train the model
    :return:
    """
    model = create_model_2nd_dim().to(device)

    X_train, y_train = load_data("data/X_train.pkl"), load_data("data/Y_train.pkl")
    X_test, y_test = load_data("data/X_test.pkl"), load_data("data/Y_test.pkl")

    # pos_weight = torch.tensor([.3, .7]).to(device)  # Increase the weight of positive samples
    loss_fn = nn.BCEWithLogitsLoss()  # Use the pos_weight in the loss function
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    # create a DataLoader for your train data
    train_dataset = TensorDataset(X_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # create a DataLoader for your test data
    test_dataset = TensorDataset(X_test, y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    train_loss_values, test_loss_values = [], []

    for t in range(NUM_EPOCHS):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        train_loss_values.append(test(model, loss_fn, train_dataloader))
        test_loss_values.append(test(model, loss_fn, test_dataloader))

    torch.save(model, "my_model.pickle")
    return train_loss_values, test_loss_values, model


def train(dataloader, model, loss_fn, optimizer):
    """
    Train the model on the training data
    :param dataloader:
    :param model:
    :param loss_fn:
    :param optimizer:
    :return:
    """
    size = len(dataloader.dataset)
    model.train()
    train_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        # print("pos_pred_percent", pos_pred_percent(pred))
        loss = loss_fn(pred, y)
        train_loss += loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return train_loss / len(dataloader)


def test(model, loss_fn, dataloader):
    """
    Test the model on the test data
    :param model:
    :param loss_fn:
    :param dataloader:
    :return:
    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    return test_loss


def plot_loss(train_loss, test_loss):
    """
    Plot the training and testing loss
    :param train_loss:
    :param test_loss:
    :return:
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=train_loss, mode='lines', name='Training Loss'))
    fig.add_trace(go.Scatter(y=test_loss, mode='lines', name='Testing Loss'))
    fig.update_layout(title='Training and Testing Loss', xaxis_title='Epochs', yaxis_title='Loss')
    fig.show()


def print_test_metrics(model):
    """
    Print the accuracy, precision, and recall of the model on the test data
    :param model:
    :return:
    """
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
    """
    Get the top 3 peptides with the highest probability of being a spike protein
    :param model:
    :return:
    """
    peptides, X = process_data.load_spike_data()
    with torch.no_grad():
        y_pred = model(torch.tensor(X).float()).squeeze()
        y_pred = y_pred[:, 1].numpy()
        top_3_indices = np.argsort(y_pred)[::-1][:3]
        for i in top_3_indices:
            print(peptides[i], y_pred[i])


def main():
    train_loss, test_loss, model = create_and_train_model()
    plot_loss(train_loss, test_loss)
    print_test_metrics(model)
    # model = torch.load("my_model.pickle")
    # get_top_3_peptides(model)


if __name__ == "__main__":
    main()
