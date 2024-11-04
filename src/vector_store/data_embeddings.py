import logging
from typing import List, Dict

import pandas as pd
import torch
from nltk.tokenize import sent_tokenize
from transformers import BartTokenizer, BartForConditionalGeneration


# Initialize the logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def summarize(
    token_ids: List[int],
    model: BartForConditionalGeneration,
    tokenizer: BartTokenizer
) -> str:
    """
    Generates a summary from the given token IDs using the provided model and tokenizer.

    Args:
        token_ids (List[int]): List of token IDs.
        model (BartForConditionalGeneration): Pretrained BART model for summarization.
        tokenizer (BartTokenizer): Corresponding tokenizer.

    Returns:
        str: Generated summary as a string.
    """
    try:
        input_ids = torch.tensor([token_ids])
        summary_ids = model.generate(
            input_ids,
            num_beams=4,
            max_length=400,
            length_penalty=2.0,
            early_stopping=True
        )
        return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    except Exception as e:
        logger.error(f"Failed to summarize due to: {str(e)}")
        return " "


def prepare_data_and_summarize(file: str, chunk_size: int) -> List[Dict]:
    bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
    """
    Reads data from an Excel file, summarizes the 'Wiki_content' column using a BART model,
    and returns a list of summarized chunks along with their metadata.

    Args:
        file (str): Path to the Excel file.
        chunk_size (int): Maximum token length for each chunk.

    Returns:
        List[Dict]: List of dictionaries containing summarized content and metadata.
    """
    if chunk_size >= 900:
        raise ValueError("Max size summarization model can handle is 1024")

    # Load dataset
    try:
        logger.info("Reading dataset from %s", file)
        dataset = pd.read_excel(file)
    except Exception as e:
        logger.error("Error reading Excel file: %s", e)
        raise

    summarized_chunks = []

    for idx, row in dataset.iterrows():
        text = row["Wiki_content"]
        print('Wiki_content')
        print(text)
        metadata = {
            "Year": row["Year"],
            "Team_Name": row["Team_Name"],
            "Wiki": row["Wiki"],
            "Region": row["Region"],
            "Location": row["Location"],
            "Institution": row["Institution"],
            "Section": row["Section"],
            "Project_Title": row["Project_Title"],
            "Track": row["Track"],
            "Abstract": row["Abstract"],
            "Parts": row["Parts"],
            "Medal": row["Medal"],
            "Nominations": row["Nominations"],
            "Awards": row["Awards"],
        }
        sentences = sent_tokenize(text)

        chunk, current_length, chunks = [], 0, []

        for sentence in sentences:
            token_ids = bart_tokenizer.encode(sentence, add_special_tokens=False)
            if current_length + len(token_ids) + 2 > chunk_size:
                if chunk:
                    chunk = [bart_tokenizer.bos_token_id] + chunk + [bart_tokenizer.eos_token_id]
                    chunks.append(chunk)
                chunk, current_length = [], 0
            chunk.extend(token_ids)
            current_length += len(token_ids)

        if chunk:
            chunk = [bart_tokenizer.bos_token_id] + chunk + [bart_tokenizer.eos_token_id]
            chunks.append(chunk)

        summaries = [summarize(chunk, bart_model, bart_tokenizer) for chunk in chunks]

        print('summaries')
        print(summaries)

        for summary in summaries:
            summarized_chunks.append({
                "Wiki_content": summary,
                "Metadata": metadata
            })

    return summarized_chunks


def mean_pooling(model_output, attention_mask):
    """Applies mean pooling to the model output using the attention mask."""
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    mask = input_mask_expanded.sum(1)
    return embeddings / torch.clamp(mask, min=1e-9)


def compute_embeddings(doc, tokenizer, model):
    """Processes a single document to compute embeddings."""
    inputs = tokenizer(doc['Wiki_content'], return_tensors='pt', padding=True, truncation=False)
    with torch.no_grad():
        outputs = model(**inputs)
    sentence_embeddings = mean_pooling(outputs, inputs['attention_mask'])
    return sentence_embeddings.squeeze().tolist()


def compute_query_embeddings(query_text, tokenizer, model):
    """Processes text to compute embeddings sequentially for queries."""
    inputs = tokenizer(query_text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    sentence_embeddings = mean_pooling(outputs, inputs['attention_mask'])
    return sentence_embeddings.squeeze().tolist()


def get_embeddings_from_chunked_text(
    file: str,
    chunk_size: int,
    embedding_model,
    embedding_tokenizer
):
    """
    Processes documents from an Excel file to compute embeddings for summarized text chunks.

    Args:
        file (str): Path to the Excel file.
        chunk_size (int): Maximum token length for each chunk during summarization.
        embedding_model: Transformer model to compute embeddings.
        embedding_tokenizer: Tokenizer compatible with the embedding model.


        """
    logger.info("Starting embedding computation from chunked text")
    chunked_docs = prepare_data_and_summarize(file, chunk_size)
    embeddings = []

    for doc in chunked_docs:
        embedding = compute_embeddings(doc, embedding_tokenizer, embedding_model)
        embeddings.append(embedding)

    logger.info("Completed embedding computation")
    return embeddings, chunked_docs
