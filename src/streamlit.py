from src.main import Chatbot
import streamlit as st
from src.vector_store.RAGConfig import RAGConfig

# Run with the best configuration from previous evaluation used to populate vector store
config = RAGConfig(
    embedding_model_name='sentence-transformers/all-MiniLM-L6-v2',
    chunk_size=250
)

bot = Chatbot(config, 'igem')


def generate_response(input: str, bot: Chatbot):
    result, _ = bot.process_input(input)
    return result


def main():
    st.title('iGEM Data Explorer')
    with st.sidebar.expander("User Guide", expanded=False):
        st.sidebar.markdown(
            """
            The iGEM Data Explorer excels in semantic search, designed to understand and process natural language queries.
             It is not optimized for direct metadata searches like specific awards or team achievements by year.
             To get the most accurate information:

            - **Use Contextual Inquiries**: Frame your questions with details likely discussed in project wiki.

            - **Incorporate Key Terms**: Include specific terms like "biosensors" or "CRISPR."

            - **Opt for Open-Ended Questions**: Such as "What projects involved CRISPR in 2019 and their objectives?"

            **Data Coverage**: Currently, the database only includes projects from the 2019 iGEM wikis as part of our proof of concept.
            """
        )

    # User-provided prompt
    user_input = st.text_input("Ask a question about 2019 iGEM projects:")

    if st.button("Submit"):
        with st.spinner("Working on it..."):
            response = generate_response(input=user_input, bot=bot)
            st.write(response)


if __name__ == '__main__':
    main()
