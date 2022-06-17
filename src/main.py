from config import Config
from src.evaluate import evaluate
from src.preprocess import process_data
from src.train import train


def main(config):
    process_data(config)
    train(config)
    evaluate(config)


if __name__ == "__main__":
    config = Config()
    main(config)
