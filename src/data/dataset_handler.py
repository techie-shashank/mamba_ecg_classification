# dataset_handler.py
class DatasetHandler:
    def __init__(self, dataset_name, config):
        self.dataset_name = dataset_name
        self.config = config
        self.handler = self._get_handler()

    def _get_handler(self):
        if self.dataset_name == "ptbxl":
            from data.ptbxl.handler import PTBXLHandler
            return PTBXLHandler(self.config)
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

    def load_data(self):
        return self.handler.load_data()

    def split_data(self, X, Y):
        return self.handler.split_data(X, Y)

    def preprocess_data(self, X, Y):
        return self.handler.preprocess_data(X, Y)

    def get_dataset(self, X, Y):
        return self.handler.get_dataset(X, Y)

    def visualize_record(self, X, idx, true_labels, preds):
        return self.handler.visualize_record(X, idx, true_labels, preds)
