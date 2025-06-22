import argparse
import logging
import torch
from data.dataset_handler import DatasetHandler
from utils import get_config_for_testing, get_model_class


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model and plot example ECGs.")
    parser.add_argument("--model", type=str, required=True, choices=["fcn", "lstm", "mamba"], help="Model type")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset to use")
    parser.add_argument("--limit", type=int, required=False, default=100, help="Limit the number of records to load")
    parser.add_argument("--idx", type=int, required=False, default=25, help="ECG Id record to evaluate and plot")
    args = parser.parse_args()

    base_dir = f"./../saved_model/{args.dataset}/{args.model}"
    config = get_config_for_testing(base_dir)
    config["limit"] = args.limit
    idx = args.idx

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)

    dataset_handler = DatasetHandler(dataset_name=args.dataset, config=config)

    logger.info(f"Loading data for dataset: {args.dataset}")
    X, Y = dataset_handler.load_data()
    logger.info("Loaded raw data: %s", X.shape)
    logger.info("Labels shape: %s", Y.shape)

    X, y = dataset_handler.preprocess_data(X, Y)
    logger.info("Data preprocessing completed.")

    logger.info("Creating DataLoader")
    dataset = dataset_handler.get_dataset(X, y)
    logger.info("DataLoader created successfully.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mlb = dataset_handler.handler.mlb
    sample_X, _ = dataset[0]
    time_steps = sample_X.shape[0]
    input_channels = sample_X.shape[1]
    num_classes = len(y[0]) if y.ndim > 1 else 2

    model_class = get_model_class(args.model)
    model = model_class(input_channels, time_steps, num_classes).to(device)
    model_path = f"./../saved_model/{args.dataset}/{args.model}/model.pt"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    criterion = torch.nn.BCEWithLogitsLoss() if num_classes > 2 else torch.nn.CrossEntropyLoss()
    is_multilabel = num_classes > 2

    # Get the single test record and label
    x_single = X[idx]
    y_single = y[idx]

    # Convert to tensor and add batch dimension
    x_tensor = torch.tensor(x_single, dtype=torch.float32).unsqueeze(0).to(device)

    # Prepare label tensor, ensure correct dtype for loss
    if is_multilabel:
        y_tensor = torch.tensor(y_single, dtype=torch.float32).unsqueeze(0).to(device)
    else:
        y_tensor = torch.tensor(y_single, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(x_tensor)
        loss = criterion(outputs, y_tensor)
        logger.info(f"Loss for record {idx}: {loss.item():.4f}")

        if is_multilabel:
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).int().cpu().numpy()
            true_labels = y_tensor.int().cpu().numpy()
        else:
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            true_labels = y_tensor.cpu().numpy()

        if is_multilabel:
            true_labels = [mlb.classes_[i] for i in range(len(true_labels[0])) if true_labels[0][i] == 1]
            preds = [mlb.classes_[i] for i in range(len(preds[0])) if preds[0][i] == 1]
        else:
            label_mapping = {i: v for i, v in enumerate(dataset_handler.handler.binary_classes)}
            true_labels = label_mapping[true_labels[0]]
            preds = label_mapping[preds[0]]

        logger.info(f"True label: {true_labels}")
        logger.info(f"Predicted label: {preds}")

    dataset_handler.visualize_record(x_single, idx, true_labels, preds)


if __name__ == "__main__":
    main()
