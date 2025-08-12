from src.utils import read_yaml
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

def run_training(config_path: str):
    config = read_yaml(config_path) if config_path else None

    ingestion = DataIngestion(config)
    train_csv, test_csv = ingestion.initiate_data_ingestion()

    transformer = DataTransformation(config)
    train_arr, test_arr, _ = transformer.initiate_data_transformation(train_csv, test_csv)

    trainer = ModelTrainer(config)
    best_model_name, r2_value = trainer.initiate_model_trainer(train_arr, test_arr)

    print(f"Best Model: {best_model_name}, R2 Score: {r2_value:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=False, help="Path to train.yaml")
    args = parser.parse_args()
    run_training(args.config)
