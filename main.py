from tavily import TavilyClient
from ollama import chat
from dotenv import load_dotenv
import os
import streamlit as st
load_dotenv()
# Initialize Tavily client
api_key_tavily = os.environ['TAVILY_API_KEY']
print(f"Tavily API Key: {api_key_tavily}")
tavily_client = TavilyClient(api_key=api_key_tavily)

# Ollama models
QUERY_GENERATION_MODEL = "llama2"
DEEP_RESEARCH_MODEL = "gemma3:27b-it-qat"
MAX_QUERY_LENGTH = 400  # Tavily's maximum query length

def generate_query(topic):
    """Generate a query using the query generation LLM for research."""
    stream = chat(
        model=QUERY_GENERATION_MODEL,
        messages=[{"role": "user", "content": f"Generate a detailed research query about: {topic}. Include keywords for comprehensive research and analysis."}],
        stream=True
    )
    response = ""
    for chunk in stream:
        response += chunk['message']['content']
    return response.strip()[:MAX_QUERY_LENGTH]  # Truncate the query to the maximum allowed length

def summarize_sources(sources):
    """Summarize the collected sources using the deep research LLM."""
    # Extract text content from each source
    combined_text = "\n\n".join(source.get("content", "") for source in sources)
    stream = chat(
        model=DEEP_RESEARCH_MODEL,
        messages=[{"role": "user", "content": f"Summarize the following information, focusing on key insights and important details:\n\n{combined_text}"}],
        stream=True
    )
    response = ""
    for chunk in stream:
        response += chunk['message']['content']
    return response.strip()

def reflect_on_summary(summary):
    """Reflect on the summary to refine insights, identifying trends, patterns, and actionable recommendations."""
    stream = chat(
        model=DEEP_RESEARCH_MODEL,
        messages=[{"role": "user", "content": f"Reflect on the following summary, identifying trends, patterns, and actionable recommendations for further research:\n\n{summary}"}],
        stream=True
    )
    response = ""
    for chunk in stream:
        response += chunk['message']['content']
    return response.strip()

def finalize_summary(refined_summaries):
    """Generate a detailed research report based on all refined summaries, including key findings and recommendations."""
    combined_summaries = "\n\n".join(refined_summaries)
    stream = chat(
        model=DEEP_RESEARCH_MODEL,
        messages=[{"role": "user", "content": f"Generate a detailed research report based on the following refined summaries, including key findings and recommendations:\n\n{combined_summaries}"}],
        stream=True
    )
    response = ""
    for chunk in stream:
        response += chunk['message']['content']
    return response.strip()

def deep_research(topic):
    """Perform deep research on a given topic."""
    print(f"Starting deep research on: {topic}")
    refined_summaries = []
    queries = [generate_query(topic) for _ in range(3)]

    for iteration, query in enumerate(queries):  # Repeat the process 3 times
        print(f"Iteration {iteration + 1} for topic: {topic}")

        # Step 1: Use pre-generated query
        print(f"Generated Query: {query}")
        
        # Step 2: Perform web research
        response = tavily_client.search(query)
        sources = response.get("results", [])
        print(f"Collected {len(sources)} sources.")
        
        # Step 3: Summarize sources
        summary = summarize_sources(sources)
        print(f"Summary:\n{summary}")
        
        # Step 4: Reflect on summary
        refined_summary = reflect_on_summary(summary)
        print(f"Refined Summary:\n{refined_summary}")
        
        # Retain the refined summary
        refined_summaries.append(refined_summary)
    
    # Step 5: Finalize the research report
    detailed_report = finalize_summary(refined_summaries)
    print(f"Detailed Research Report for {topic}:\n{detailed_report}")
    
    return {
        "query": query,
        "sources": sources,
        "summary": summary,
        "refined_summary": refined_summary,
        "detailed_report": detailed_report
    }

# Streamlit UI
st.title("Ollama Deep Researcher")

user_query = st.text_input("Enter your research query:")

if st.button("Start Research"):
    research_output = deep_research(user_query)   # Perform deep research

    st.subheader("Generated Query:")
    st.write(research_output['query'])

    st.subheader("Collected Sources:")
    for source in research_output['sources']:
        st.write(f"- {source.get('title', 'No Title')}: {source.get('url', 'No URL')}")

    st.subheader("Summary:")
    st.write(research_output['summary'])

    st.subheader("Refined Summary:")
    st.write(research_output['refined_summary'])

    st.subheader("Detailed Research Report:")
    st.write(research_output['detailed_report'])
