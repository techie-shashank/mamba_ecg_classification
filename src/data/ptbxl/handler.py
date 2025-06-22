import numpy as np
import matplotlib.pyplot as plt


def plot_ecg(signal, fs=100, title="ECG Signal"):
    if signal.shape[0] < signal.shape[1]:
        signal = signal.T

    time = np.arange(signal.shape[0]) / fs
    num_leads = signal.shape[1]

    plt.figure(figsize=(10, num_leads * 2))
    for i in range(num_leads):
        plt.subplot(num_leads, 1, i + 1)
        plt.plot(time, signal[:, i], label=f"Lead-{i+1}")
        plt.ylabel("mV")
        plt.legend(loc='upper right')
        if i == 0:
            plt.title(title)
    plt.xlabel("Time (s)")
    plt.tight_layout()
    plt.show()


class PTBXLHandler:
    def __init__(self, config):
        self.config = config
        self.mlb = None  # MultiLabelBinarizer instance, if needed
        self.binary_classes = None

    def load_data(self):
        from data.ptbxl import loaders
        return loaders.load_data(
            data_dir=self.config["data_dir"],
            sampling_rate=self.config["sampling_rate"],
            limit=self.config.get("limit")
        )

    def split_data(self, X, Y):
        from data.ptbxl import utils
        return utils.split_train_test(X, Y)

    def preprocess_data(self, X, Y, mlb=None):
        from data.ptbxl import preprocessing
        is_multilabel = self.config.get("is_multilabel", False)
        if is_multilabel:
            X, Y, mlb = preprocessing.preprocess_data(X, Y, self.mlb)
            if self.mlb is None:
                self.mlb = mlb
        else:
            X, Y, self.binary_classes = preprocessing.preprocess_data_binary(X, Y)
        return X, Y

    def get_dataset(self, X, Y):
        from data.ptbxl.dataset import PTBXL
        return PTBXL(X, Y)

    def visualize_record(self, X, idx, true_labels, preds):
        plot_ecg(X, title=f"Test Record {idx} - True: {true_labels} Pred: {preds}")