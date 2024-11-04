#!/bin/bash
export PYTHONPATH="$(pwd):$PYTHONPATH"

if [ "$APP_MODE" == "evaluation" ]; then
    echo "Initializing database for evaluation..."
    python src/evaluation/benchmark.py
elif [ "$APP_MODE" == "production" ]; then
    echo "Checking if the database needs initialization..."
    # Run the database initialization script before starting the Streamlit app
    python src/init_qdrant_for_prod.py

    echo "Starting the Streamlit application..."
    streamlit run src/streamlit.py
else
    echo "Error: APP_MODE not set to 'evaluation' or 'production'."
    exit 1
fi
