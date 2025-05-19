from fastapi import FastAPI, HTTPException
from tavily import TavilyClient
import openai
from dotenv import load_dotenv
import os
from typing import List, Dict, Any, Optional
import time
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Initialize clients
tavily_client = TavilyClient(api_key=os.environ['TAVILY_API_KEY'])
openai.api_key = os.environ.get('OPENAI_API_KEY')

# Configuration
# === User-editable configuration ===
QUERY_GENERATION_MODEL = "gpt-4o"   # Model for query generation
DEEP_RESEARCH_MODEL = "gpt-4o"      # Model for summarization/gap analysis
MAX_QUERY_LENGTH = 400              # Max length for generated queries
MAX_RETRIES = 3                     # Number of retries for web search
DELAY_BETWEEN_REQUESTS = 2          # Delay (seconds) between API calls
MAX_ITERATIONS = 3                  # Number of research/refinement iterations
# ===================================

app = FastAPI(title="Iterative Market Research API",
              description="API for performing deep, iterative market research using AI-powered analysis")

class ResearchRequest(BaseModel):
    domain: str
    company_name: Optional[str] = Field(None, description="Specific company to focus on")
    metrics: Optional[List[str]] = None
    custom_operator: Optional[str] = None

class ResearchResponse(BaseModel):
    final_analysis: str
    iterations: List[Dict[str, Any]]
    all_sources: List[Dict[str, Any]]

def generate_initial_query(domain: str, company_name: str = None, metrics: List[str] = None, custom_operator: str = None) -> str:
    """Generate initial comprehensive research query"""
    prompt = f"""
    Create a comprehensive web search query for market research about: {domain}
    {f"focusing on company: {company_name}" if company_name else ""}
    {f"analyzing metrics: {', '.join(metrics)}" if metrics else ""}
    {f"using analysis method: {custom_operator}" if custom_operator else ""}

    The query should:
    - Be specific enough to get relevant results
    - Include important industry keywords
    - Cover both broad trends and specific details
    - Be under {MAX_QUERY_LENGTH} characters
    """
    
    response = openai.chat.completions.create(
        model=QUERY_GENERATION_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=200,
    )
    return response.choices[0].message.content.strip()

def generate_refinement_query(domain: str, previous_summary: str, knowledge_gaps: List[str]) -> str:
    """Generate refined search query based on identified gaps"""
    prompt = f"""
    Based on the following research summary and identified knowledge gaps about {domain},
    create a refined web search query that specifically addresses these gaps:
    
    Previous Summary:
    {previous_summary}
    
    Knowledge Gaps:
    {', '.join(knowledge_gaps)}
    
    The new query should:
    - Target the missing information specifically
    - Use precise terminology
    - Be under {MAX_QUERY_LENGTH} characters
    """
    
    response = openai.chat.completions.create(
        model=QUERY_GENERATION_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,  # Slightly more creative for gap filling
        max_tokens=200,
    )
    return response.choices[0].message.content.strip()

def identify_knowledge_gaps(domain: str, summary: str) -> List[str]:
    """Analyze summary to identify missing information"""
    prompt = f"""
    Analyze this market research summary about {domain} and identify the 3 most important
    knowledge gaps or unanswered questions that would improve the research quality.
    
    Focus on:
    - Missing data points
    - Unclear trends
    - Lack of specific examples
    - Areas needing more depth
    
    Summary:
    {summary}
    
    Return only a bulleted list of the key gaps, nothing else.
    """
    
    response = openai.chat.completions.create(
        model=DEEP_RESEARCH_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=200,
    )
    return [line.strip('- ').strip() for line in response.choices[0].message.content.split('\n') if line.strip()]

def summarize_results(sources: List[Dict[str, Any]], domain: str, metrics: List[str] = None) -> str:
    """Create comprehensive summary from search results"""
    combined_text = "\n\n".join(
        f"Source {i+1}: {source.get('title', 'Untitled')}\n"
        f"URL: {source.get('url', 'No URL')}\n"
        f"Content: {source.get('content', 'No content')[:2000]}..."  # Limit content length
        for i, source in enumerate(sources)
    )

    metrics_context = f" focusing on {', '.join(metrics)}" if metrics else ""

    prompt = f"""
    Synthesize a comprehensive summary from these search results about {domain}{metrics_context}:

    {combined_text}

    Include:
    1. Key findings and statistics
    2. Trends and patterns
    3. Conflicting information
    4. Notable missing information

    Organize the summary clearly with headings.
    """

    response = openai.chat.completions.create(
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

# Remove the hardcoded query and research output
# user_query = """Conduct a comprehensive market research analysis on APAR Industries Limited, focusing on its current market share, product portfolio (especially in conductors, specialty oils, and cables), and competitive positioning in India and international markets. Analyze historical revenue growth, segment-wise performance, key customers, and strategic partnerships. Compare APAR's technological innovation, pricing strategy, and supply chain resilience against key competitors such as Polycab, Sterlite Power, and Bharat Bijlee. Assess emerging trends in power transmission, renewable energy integration, and electric mobility that could influence APAR’s growth trajectory over the next 5 years. Include risk factors (e.g., raw material prices, regulatory changes), and identify market opportunities or untapped geographies for expansion. Provide data-backed insights and recommendations"""
# research_output = deep_research(user_query)   # Perform deep research
# print(f"Detailed Research Report:\n{research_output['detailed_report']}")