# Ollama Deep Researcher

This project performs deep research on a given topic using Ollama for language model inference and Tavily for web search.

## Setup

1.  **Install dependencies:**

    ```bash
    pip install tavily ollama python-dotenv streamlit
    ```

2.  **Set up environment variables:**
    *   Create a `.env` file in the project directory.
    *   Add your Tavily API key to the `.env` file:

        ```
        TAVILY_API_KEY=YOUR_TAVILY_API_KEY
        ```

3.  **Run the Streamlit application:**

    ```bash
    streamlit run main.py
    ```

## Components

*   `main.py`: This file contains the main application logic, including:
    *   `generate_query`: Generates a research query using a language model.
    *   `summarize_sources`: Summarizes the content from multiple web sources using a language model.
    *   `reflect_on_summary`: Refines the summary by identifying trends and patterns.
    *   `finalize_summary`: Generates a detailed research report based on the refined summaries.
    *   `deep_research`: Orchestrates the entire research process.
    *   Streamlit UI: Provides a user interface for entering a research query and viewing the results.

## Usage

1.  Enter a research query in the text input field.
2.  Click the "Start Research" button.
3.  The application will perform deep research on the topic and display the generated query, collected sources, summary, refined summary, and detailed research report in the Streamlit UI.
