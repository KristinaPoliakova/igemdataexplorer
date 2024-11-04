import random
from groq import Groq, RateLimitError
import os
import logging
from typing import List, Dict, Any


# Initialize the logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def call_llm_for_text_generation(prompt_in_chat_format: List[Dict[str, str]]) -> str:
    api_key = os.getenv('GROQ_API_KEY')
    groqClient = Groq(api_key=api_key)
    try:
        chat_completion = groqClient.chat.completions.create(
            messages=prompt_in_chat_format,
            model="llama3-70b-8192",
            temperature=0,
        )
        return chat_completion.choices[0].message.content
    except RateLimitError as e:
        logger.info(e)


def generate_question_answers(generations: int, prepared_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    results = []
    # Ensure unique selections of contexts by adjusting the sample size if necessary
    sample_size = min(generations, len(prepared_data))
    sampled_contexts = random.sample(prepared_data, k=sample_size)

    for sampled_context in sampled_contexts:
        print(sampled_context['Wiki_content'])
        prompt_in_chat_format = [
            {
                "role": "system",
                "content": (
                    "You are an expert assistant specializing in past iGEM projects. Your task is to:\n"
                    "1. Read the provided context, which may include information about various past iGEM teams and "
                    "their projects. \n"
                    "2. Generate a **broad question** that helps discover whether any iGEM team has worked on a "
                    "specific idea, method, or technology (e.g., use of cyan fluorescent protein).\n"
                    "3. Provide a corresponding answer based solely on the provided context.\n\n"
                    "Requirements:\n"
                    "- The question should be more general and aimed at identifying if any team has worked on the "
                    "specified topic or tackled a specific problem.\n"
                    "- Frame the question as an inquiry that could help identify similar projects or approaches used "
                    "by different teams. \n"
                    "Formatting:\n"
                    "Output your response exactly in the following format, without adding anything else:\n"
                    "Question: [insert your question here]\n"
                    "Answer: [insert your answer here]"
                )
            },
            {
                "role": "user",
                "content": f"Context:\n{sampled_context['Wiki_content']}"
            }
        ]

        try:
            output_qa_couple = call_llm_for_text_generation(prompt_in_chat_format)
            question = output_qa_couple.split("Question: ")[-1].split("Answer: ")[0].strip()
            answer = output_qa_couple.split("Answer: ")[-1].strip()

            results.append(
                {
                    "qa_context": sampled_context['Wiki_content'],
                    "question": question,
                    "reference_answer": answer,
                    "qa_meta": sampled_context['Metadata']
                }
            )
        except Exception as e:
            print(f"Failed to process context {sampled_context['Wiki_content'][:50]} due to an error: {e}")
            continue

    return results
