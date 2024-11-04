import glob
import json

import numpy as np
import pandas as pd
import logging

# Initialize the logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def find_best_configuration():
    outputs = [
        pd.DataFrame(json.load(open(file))).assign(settings=file)
        for file in glob.glob("src/evaluation/output/output/*.json")
    ]

    if not outputs:
        logger.warning("No data to analyze for configurations.")
        return None

    combined_results = pd.concat(outputs)
    if "eval_score" not in combined_results.columns:
        logger.warning("Evaluation scores are missing from the data.")
        return None

    combined_results['normalized_eval_score'] = combined_results['eval_score'].replace(0, np.nan)
    combined_results['normalized_eval_score'] = (combined_results['normalized_eval_score'] - 1) / 4

    average_scores = combined_results.groupby("settings")["normalized_eval_score"].mean()
    best_configuration_path = average_scores.idxmax()
    best_score = average_scores.max()
    logger.info(f"Best configuration: {best_configuration_path} with an average score of {best_score:.4f}")

    # Extract the best test settings
    best_file = combined_results.loc[combined_results["settings"] == best_configuration_path]
    best_test_settings = best_file["test_settings"].values[0]  # Assuming test_settings is a string

    # Parse the RAGConfig from the test_settings string
    try:
        settings = best_test_settings.strip("RAGConfig(").rstrip(")")
        settings_dict = dict(item.strip().split("=") for item in settings.split(", "))
        dataset = settings_dict['dataset']
        chunk_size = int(settings_dict['chunk_size'])
        embedding_model = settings_dict['embedding_model_name']
        return dataset, chunk_size, embedding_model, best_score
    except Exception as e:
        logger.error(f"Error parsing test settings: {e}")
        return None
