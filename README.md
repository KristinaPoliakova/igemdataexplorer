# Welcome to iGEM Data Explorer!

## Online version
If you want to start with having a look at a running application, it is available [here](https://explorer.igem-munich.com/)!

## Scope and Expansion Potential  
The current version of the iGEM Data Explorer includes data exclusively from the 2019 iGEM wikis as part of our proof of concept. This specific year was chosen to demonstrate the capabilities of the RAG system in a controlled environment. However, the architecture and design of our tool are built with scalability in mind. It is fully capable of expanding to include all wikis from all years of the iGEM competition. This expansion can be seamlessly integrated as the project scales, providing an ever-growing repository of iGEM projects for research and analysis.

## Overview
The iGEM Data Explorer is a sophisticated tool designed to empower researchers and enthusiasts by facilitating swift and efficient access to the expansive archives of iGEM wikis. Our software utilizes a dynamic and robust Retrieval-Augmented Generation (RAG) architecture, allowing users to navigate and leverage the wealth of iGEM's collective knowledge seamlessly.  
For those interested in a deeper dive into how RAG works and its implications for enhancing large language models, we recommend visiting this [comprehensive guide on RAG](https://www.superannotate.com/blog/rag-explained). This resource offers detailed insights into the technology behind RAG, exploring its foundational concepts and practical applications in real-world scenarios.  
The iGEM Data Explorer leverages the RAG system to efficiently retrieve relevant information from an extensive archive of iGEM wikis. Designed for the synthetic biology community, this tool streamlines research by providing quick access to vast amounts of data from past iGEM projects. It uses a sophisticated data processing pipeline that involves scraping wiki data, summarizing scraped data, embedding it for semantic search, and retrieving the most relevant content in response to user queries.

## Features
- **Pioneering Interactive Wiki Queries**: As the first tool of its kind in the iGEM community, the iGEM Data Explorer revolutionizes how researchers access historical project data. Say goodbye to manually searching through team wikis. Now, just ask and receive instant, relevant answers directly through our application. ⚡ 
- **User-Centric Query Interface**: Simple, user-friendly querying interface to fetch relevant data quickly 🔍
- **Robust Documentation and Logging**: Benefit from a fully documented, thoroughly commented codebase that ensures ease of use, maintenance, and scaling. Paired with our detailed logging system, you can troubleshoot with confidence and clarity. 📚  
- **LLM Validation**: Our system uses large language model as a 'judge' to evaluate and optimize various Retrieval-Augmented Generation (RAG) settings. This ensures the selection of the most effective configuration for accuracy and efficiency before deployment. 🤖✓

## Application Modes

The application can be run in two different modes: `production` and `evaluation`. These modes are designed to cater to different phases of the application lifecycle and user needs.

### Production Mode

- **Purpose**: The `production` mode is optimized for performance and stability during regular operation such as querieng the vector store with user query.

### Evaluation Mode

- **Purpose**: The `evaluation` mode is intended for RAG performance evaluation based on different RAG settings. The best setting is then used to initialize qdrant and compute data embeddings in production mode if starting from scratch. 

### Setting the Mode

You can specify which mode to run by setting the `APP_MODE` environment variable in the `.env` file (s. 'Configure Environment Variables' below).

## Project Installation Guide

Welcome to the installation guide for our application. This document provides step-by-step instructions on how to set up and run the application using Docker.

## Prerequisites

Before you begin, ensure you have the following installed:
- **Docker**: Download and install Docker from [Docker's official website](https://docs.docker.com/get-docker/).
- **Docker Compose**: If Docker was not installed using the link provided above, Docker Compose might need to be installed separately. You can find the installation instructions on [Docker's official website](https://docs.docker.com/compose/install/)
- **Git**: Required for cloning the repository. Install Git from [Git's official website](https://git-scm.com/downloads).
- **Groq API key**: Required to prompt LLM with processed and aggregated wiki contents. Get it from [Groq's official website](https://console.groq.com/keys).
To do this, follow these steps:
1. Visit the link provided and log in.
2. Navigate to the API keys section as shown here:
![](https://static.igem.wiki/teams/5102/software/api-key.png)
3. Click 'Create API key'. You will need to provide a name for the key.
4. Once the key is created, it can be copied and used as described in the setup steps under 'Configure Environment Variables':
![](https://static.igem.wiki/teams/5102/software/create-api-key.png)

#### 1. Clone the Repository

Clone the project repository to your local machine using terminal ([guide](https://www.freecodecamp.org/news/command-line-for-beginners/) on how to use terminal):

```bash
git clone git@gitlab.igem.org:2024/software-tools/munich.git
cd munich
```
### 2. Configure Environment Variables

Before running the application, set up the required environment variables. You can define these in a `.env.example` file already located in the same directory as `docker-compose.yml`. For this, set the following variables:

```plaintext
GROQ_API_KEY=your_groq_api_key_here
APP_MODE=production  # Change to 'evaluation' as needed
```
Replace `your_groq_api_key_here` with the actual key that you obtained during the 'Prerequisites' step.  
Choose whether the application should run in production or evaluation mode by setting the `APP_MODE` appropriately.  
Your `.env` should look like this:  

![](https://static.igem.wiki/teams/5102/software/new-env.png)

### 3. Start the Application Using Docker Compose
Once you have installed all the prerequisites and set up your environment, you can start the application using Docker Compose. This will ensure that all components of the application, including the chat application and the Qdrant vector database, are correctly orchestrated and run in unison.  
Follow these steps to launch the application:
- Navigate to Your Project Directory: Open a terminal and change to the directory where you have cloned your project. This directory should contain the docker-compose.yml file.
- Initial Build and Start: Run the following command to build and start all services defined in your docker-compose.yml:
```bash
docker compose up --build
```
- Subsequent Starts:  For subsequent starts, you can skip the build process unless you make changes to the Docker configuration. Use the following command to quickly start all services:  
```bash
docker compose up
```

## Accessing the Application
Once the containers are up, you can access the chat application by navigating to `http://localhost:8501/` from your browser. This opens the user interface of your application hosted on the specified port. 

## Monitoring Qdrant Service
The Qdrant vector database can be accessed through its HTTP API at http://localhost:6333 for direct API interactions.

## Stopping the Application
To stop the application, run:

```bash
docker compose down
```

## Future Directions
As we look ahead, the iGEM Data Explorer is poised for several enhancements that aim to refine functionality and expand capabilities, addressing both current limitations and future technological advancements. Here are the key areas of focus:
- **Improved Data Scraping and Cleaning**: To enhance the quality and usability of the data, we will advance our web scraping and data cleaning techniques. By retaining more contextual information, such as references and laboratory data, and improving the way we chunk text, the system will be able to provide richer and more accurate outputs.
- **Enhanced Metadata Filtering**: Currently, our search capabilities are primarily based on wiki content. Plans are underway to integrate more sophisticated natural language processing techniques to process user queries. This will allow us to apply filters that combine metadata from the Phoenix project with similarity search, providing more targeted and relevant search results.
- **Source Tracking and Display**: We intend to implement a feature that records and displays the source webpage of each data chunk within the search results. This will provide users with direct access to the original content for deeper investigation and verification.
 
