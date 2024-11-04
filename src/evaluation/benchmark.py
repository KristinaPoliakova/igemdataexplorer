import os
from collections import defaultdict
import time

import numpy as np

from qa_utils import generate_question_answers, call_llm_for_text_generation
from src.vector_store.RAGConfig import RAGConfig
from src.vector_store.data_embeddings import prepare_data_and_summarize
from src.main import Chatbot
from src.vector_store.init_qdrant import init_qdrant
import glob
import json
import pandas as pd
import logging
from matplotlib import pyplot as plt

# Initialize the logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def prepare_eval_datasets_for_diff_chunk_sizes(chunk_sizes, dataset_path, generations):
    eval_datasets = {}
    for chunk_size in chunk_sizes:
        start_time = time.time()
        logger.info(f"Preparing data for chunk size {chunk_size}")
        prepared_data = prepare_data_and_summarize(dataset_path, chunk_size)
        logger.info(f"Generating question-answer pairs for chunk size {chunk_size}")
        qa_pairs = generate_question_answers(generations, prepared_data)
        eval_datasets[chunk_size] = {'qa_pairs': qa_pairs, 'prepared_data': prepared_data}
        logger.info(f"generated question answers {eval_datasets[chunk_size]}")
        end_time = time.time()
        logger.info(f"Data prepared and question answers generated for chunk size {chunk_size} in {end_time - start_time:.2f} seconds")

    return eval_datasets


def run_rag_tests(
        bot: Chatbot,
        config: RAGConfig,
        eval_dataset,
        output_file: str,
):
    try:  # load previous generations if they exist
        with open(output_file, "r") as file:
            logger.info(f'Loading previous results from {output_file}')
            outputs = json.load(file)
    except FileNotFoundError:
        logger.info(f'No previous results found, starting fresh.')
        outputs = []

    for example in eval_dataset:
        question = example["question"]
        if question in [output["question"] for output in outputs]:
            continue
        answer, relevant_docs = bot.process_input(question)
        result = {
            "question": question,
            "reference_answer": example['reference_answer'],
            "qa_context": example['qa_context'],
            "qa_meta": example['qa_meta'],
            "answer": str(answer),
            "qdrant_context": relevant_docs,
        }
        if config:
            result["test_settings"] = str(config)
        outputs.append(result)
        logger.info(f'Processed question: {question}')

    # Move the file writing operation outside the loop
    with open(output_file, "w") as file:
        json.dump(outputs, file, indent=1)
        logger.info(f'Saved results to {output_file}')


def evaluate_answers(answer_path: str) -> None:
    if os.path.isfile(answer_path):
        with open(answer_path, "r") as file:
            answers = json.load(file)
            logger.info(f'Loaded answers from {answer_path}')
    else:
        logger.error(f"File not found: {answer_path}")
        return

    for experiment in answers:
        eval_prompt = [
            {
                "role": "system",
                "content": (
                    "You are an expert evaluator of iGEM project responses. Your task is to:\n"
                    "1. Read the provided question, response, and reference answer.\n"
                    "2. Evaluate the response based on its correctness, accuracy,"
                    " and factuality concerning the question and the reference answer.\n"
                    "3. Assign a numerical score from 1 to 5 based on the following criteria:\n"
                    "   - **5**: The response is completely accurate, relevant, and comprehensive."
                    " It fully answers the question and aligns perfectly with the reference answer.\n"
                    "   - **4**: The response is mostly accurate and relevant, with minor omissions or errors.\n"
                    "   - **3**: The response is partially accurate but lacks important details"
                    " or contains some inaccuracies.\n"
                    "   - **2**: The response is mostly inaccurate or irrelevant, with significant errors"
                    " or omissions.\n"
                    "   - **1**: The response is incorrect, irrelevant, or does not answer the question at all.\n\n"
                    "Formatting Instructions:\n"
                    "Provide your evaluation exactly in the following format and do not include anything else:\n"
                    "Score: [insert your score here]"
                )
            },
            {
                "role": "user",
                "content": (
                    "Please evaluate the following question-answer pair:\n\n"
                    f"Question: {experiment['question']}\n"
                    f"Response: {experiment['answer']}\n"
                    f"Reference answer: {experiment['reference_answer']}\n\n"
                    "Please provide your response in the format mentioned above and do NOT add anything else to it."
                )
            }
        ]
        print('-------------------eval_prompt-------------------')
        print(eval_prompt)

        eval_result = call_llm_for_text_generation(eval_prompt)
        print('-------------------eval_result-------------------')
        print(eval_result)

        score = int(eval_result.split("Score:")[1].strip().split()[0]) if "Score:" in eval_result else 0

        experiment["eval_score"] = score

        with open(answer_path, "w") as file:
            json.dump(answers, file, indent=1)
            logger.info(f'Updated and saved evaluated answers to {answer_path}')


def benchmark_rag(configurations: list[RAGConfig], eval_datasets):
    os.makedirs("src/evaluation/output", exist_ok=True)

    # Group configurations by chunk size
    configs_by_chunk_size = defaultdict(list)
    for config in configurations:
        configs_by_chunk_size[config.chunk_size].append(config)

    for chunk_size, configs in configs_by_chunk_size.items():
        eval_dataset = eval_datasets[chunk_size]
        logger.info(f"Processing configurations for chunk size {chunk_size}")

        for config in configs:
            logger.info(f'Initializing Qdrant with configuration: chunk_size={config.chunk_size},'
                        f' embedding_model={config.embedding_model_name}')

            init_qdrant(config=config, mode='evaluation', prepared_data=eval_dataset["prepared_data"])

            logger.info(f'Sample qa document: {eval_dataset["qa_pairs"][0]}')

            settings_name = f"chunk_size:{config.chunk_size}_embeddings:{config.embedding_model_name.replace('/', '-')}"
            output_file_name = f"src/evaluation/output/rag_{settings_name}.json"
            logger.info(f"Running evaluation for {settings_name}:")

            # Initialize the chatbot with current configuration
            bot = Chatbot(config, 'igem_eval')

            # Running RAG tests
            logger.info("Running RAG tests...")
            run_rag_tests(
                bot=bot,
                config=config,
                eval_dataset=eval_dataset['qa_pairs'],
                output_file=output_file_name,
            )

            # Running evaluation
            logger.info("Running evaluation...")
            evaluate_answers(
                answer_path=output_file_name
            )


def plot():
    def simplify_model_name(setting):
        parts = setting.split('_')
        chunk_size = parts[2].split(':')[1]
        model_name = parts[3].split(':')[1].split('.')[0]
        return f"{chunk_size},{model_name}"

    outputs = [
        pd.DataFrame(json.load(open(file))).assign(settings=file)
        for file in glob.glob("src/evaluation/output/output/*.json")
    ]

    if outputs:
        result = pd.concat(outputs)
        result['settings_name'] = result['settings'].apply(simplify_model_name)
        # Replace zeros with NaN to indicate missing scores
        result['eval_score'] = result['eval_score'].replace(0, np.nan)
        result['eval_score'] = (result['eval_score'] - 1) / 4  # Normalize scores
        average_scores = result.groupby('settings_name')['eval_score'].mean().sort_values()

        # Plotting with adjustments
        plt.figure(figsize=(12, 6))
        ax = average_scores.plot(
            kind='bar',
            title='Average Evaluation Scores by Settings'
        )
        ax.set_xlabel('Settings Name')
        ax.set_ylabel('Normalized Eval Score')
        # Rotate x-axis labels vertically and adjust font size
        plt.xticks(rotation=45, ha='right', fontsize=8)
        # Adjust layout to make room for the rotated labels
        plt.tight_layout()
        # Save the plot
        plt.savefig('src/evaluation/output/output/average_scores.png')
        plt.close()


if __name__ == '__main__':
    logger.info("Starting benchmark RAG...")
    generations = 30
    chunk_sizes = [200, 300, 500, 800]
    embedding_models_names = ['sentence-transformers/all-MiniLM-L6-v2',
                               'sentence-transformers/all-mpnet-base-v2',
                               'thenlper/gte-small']

    dataset_path = 'src/data/processed_data.xlsx'

    # Create configurations
    configurations = [
        RAGConfig(chunk_size=chunk_size, embedding_model_name=embedding_model_name)
        for chunk_size in chunk_sizes
        for embedding_model_name in embedding_models_names
    ]
    eval_datasets = prepare_eval_datasets_for_diff_chunk_sizes(chunk_sizes, dataset_path, generations)

    benchmark_rag(configurations, eval_datasets)
    logger.info("Benchmark RAG completed.")

    plot()
