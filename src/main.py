from qdrant_client import QdrantClient
from groq import Groq, GroqError

from src.vector_store.data_embeddings import compute_query_embeddings
import os
from src.vector_store.RAGConfig import RAGConfig
#import anthropic


class Chatbot:
    def __init__(self,
                 config: RAGConfig,
                 collection_name: str) -> None:
        api_key = os.getenv('GROQ_API_KEY')
        if api_key is None:
            raise ValueError("GROQ_API_KEY environment variable is not set")
        self.tokenizer = config.tokenizer
        self.model = config.embedding_model
        self.qdrantClient = QdrantClient(host='qdrant', port=6333)
        self.groqClient = Groq(api_key=api_key)
        self.collection_name = collection_name

    def encode(self, query: str) -> list[float]:
        # Calculate query embedding
        query_embedding = compute_query_embeddings(query, self.tokenizer, self.model)
        return query_embedding

    def process_input(self, user_input: str) -> tuple[str, list[dict]]:
        # Encode user input and perform a search
        query_vector = self.encode(user_input)
        hits = self.qdrantClient.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=10,
            with_payload=True
        )
        # Extract payloads from hits
        retrieved_data = [hit.payload for hit in hits]
        # Construct a prompt using the retrieved data
        enriched_rag_prompt = self.construct_rag_prompt(user_input, retrieved_data)
        print(enriched_rag_prompt)
        # Query the model and return the response along with the retrieved data
        response = self.query_groq(enriched_rag_prompt)
        return response, retrieved_data

    def query_groq(self, prompt_in_chat_format: list) -> str:
        try:
            # Create a chat completion with the correct message format
            chat_completion = self.groqClient.chat.completions.create(
                messages=prompt_in_chat_format,
                model="llama3-70b-8192"
            )
            return chat_completion.choices[0].message.content
        except GroqError as e:
            return str(e)

    # use to prompt claude-3-5 instead of llama3
    # def query_groq_with_anthropic(self, prompt_in_chat_format: list):
    #     client = anthropic.Anthropic(
    #         api_key=os.environ.get("ANTHROPIC_API_KEY"),
    #     )
    #
    #     # Create the message using the Anthropics API
    #     message = client.messages.create(
    #         model="claude-3-5-sonnet-20240620",
    #         max_tokens=1024,
    #         system=prompt_in_chat_format[0]["content"],  # Directly pass the system content as a string
    #         messages=[{
    #             "role": "user",
    #             "content": prompt_in_chat_format[1]["content"]
    #         }]  # Construct the user message as required by the API
    #     )
    #     # Extract the text from the message.content
    #     try:
    #         # Ensure that message.content is a list and has at least one element
    #         if isinstance(message.content, list) and len(message.content) > 0:
    #             text_block = message.content[0]
    #             # Ensure that the first element has a 'text' attribute
    #             if hasattr(text_block, 'text'):
    #                 return text_block.text
    #             else:
    #                 print("Error: The first element in message.content does not have a 'text' attribute.")
    #                 return None
    #         else:
    #             print("Error: message.content is not a list or is empty.")
    #             return None
    #     except AttributeError as ae:
    #         print(f"AttributeError while extracting text: {ae}")
    #         return None
    #     except Exception as ex:
    #         print(f"Unexpected error while extracting text: {ex}")
    #         return None
    #

    def construct_rag_prompt(self, user_query: str, retrieved_projects: list):
        if not retrieved_projects:
            return f"User Query: {user_query} \n No relevant iGEM project information found."
        else:
            # Construct context from the retrieved projects
            context = "Retrieved Projects Information:\n"
            for project in retrieved_projects:
                context += f"Wiki_content: {project['Wiki_content']}"\
                           f"Year: {project['Year']}\n" \
                           f"Team_Name: {project['Team_Name']}\n" \
                           f"Project_Title: {project['Project_Title']}\n" \
                           f"Wiki: {project['Wiki']}\n" \
                           f"Region: {project['Region']}\n" \
                           f"Location: {project['Location']}\n" \
                           f"Institution: {project['Institution']}\n" \
                           f"Section: {project['Section']}\n" \
                           f"Track: {project['Track']}\n" \
                           f"Abstract: {project['Abstract']}\n" \
                           f"Parts: {project['Parts']}\n" \
                           f"Medal: {project['Medal']}\n" \
                           f"Nominations: {project['Nominations']}\n" \
                           f"Awards: {project['Awards']}" \

            # Construct the chat format prompt
            prompt_in_chat_format = [
                {
                    "role": "system",
                    "content": "You are a precise information retrieval assistant. Your task is to answer questions "
                               "based ONLY on information in the provided context. Follow these "
                               "rules strictly:\n"
                               "1. Use ONLY information explicitly stated in the context.\n"
                               "2. Do NOT make inferences or assumptions beyond what is directly stated.\n"
                               "3. Do not use external knowledge or information not present in the given context.\n"
                },
                {
                    "role": "user",
                    "content": f"Context:\n"
                               f"{context}\n"
                               f"---\n"
                               f"Based strictly on the above context, answer the following "
                               f"Question:\n"
                               f"{user_query}"
                },
            ]
            # Format the prompt for the chat template
            return prompt_in_chat_format
