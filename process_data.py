import numpy as np
from sklearn.model_selection import train_test_split
import pickle

NUM_AMINO_ACIDS = 20
PEPTIDE_LEN = 9


def amino_to_vec(amino: str):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    aa_to_index = {aa: i for i, aa in enumerate(amino_acids)}

    one_hot_vector = np.zeros(NUM_AMINO_ACIDS)
    one_hot_vector[aa_to_index[amino]] = 1
    return one_hot_vector


def peptide_to_input_vec(peptide: str):
    if len(peptide) != PEPTIDE_LEN:
        raise ValueError("Wrong peptide length")
    return np.concatenate([amino_to_vec(amino) for amino in peptide])


def load_data_single_file(filepath, flag):
    with open(filepath) as file:
        peptides = file.read().splitlines()
        X = np.array([peptide_to_input_vec(peptide) for peptide in peptides])
    Y = np.array([[1, 0]] * len(X)) if flag else np.array([[0, 1]] * len(X))
    # convert numpy arrays to float
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)
    return X, Y


POS_FILE = "data/pos_A0201.txt"
NEG_FILE = "data/neg_A0201.txt"


def load_both_files():
    X_pos, Y_pos = load_data_single_file(POS_FILE, True)
    X_neg, Y_neg = load_data_single_file(NEG_FILE, False)
    X = np.concatenate([X_pos, X_neg])
    Y = np.concatenate([Y_pos, Y_neg])
    return X, Y


def split_data(X, Y):
    return train_test_split(X, Y, test_size=0.1, random_state=42)


def load_spike_data():
    with open("data/spike.txt") as spike_file:
        file_text = spike_file.read()

    peptides = []
    for i in range(len(file_text) - PEPTIDE_LEN):
        peptides.append(file_text[i: i + PEPTIDE_LEN])

    X = np.array([peptide_to_input_vec(peptide) for peptide in peptides])
    X = X.astype(np.float32)
    return peptides, X


def main():
    X, Y = load_both_files()
    X_train, X_test, Y_train, Y_test = split_data(X, Y)
    # pickle the files
    with open("data/X_train.pkl", "wb") as file:
        pickle.dump(X_train, file)
    with open("data/X_test.pkl", "wb") as file:
        pickle.dump(X_test, file)
    with open("data/Y_train.pkl", "wb") as file:
        pickle.dump(Y_train, file)
    with open("data/Y_test.pkl", "wb") as file:
        pickle.dump(Y_test, file)


if __name__ == "__main__":
    main()
