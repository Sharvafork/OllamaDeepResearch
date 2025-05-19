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
QUERY_GENERATION_MODEL = "gpt-4o"
DEEP_RESEARCH_MODEL = "gpt-4o"
MAX_QUERY_LENGTH = 400
MAX_RETRIES = 3
DELAY_BETWEEN_REQUESTS = 2
MAX_ITERATIONS = 3

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
        temperature=0.5,
        max_tokens=200,
    )
    return response.choices[0].message.content.strip()

def identify_knowledge_gaps(domain: str, summary: str) -> List[str]:
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
    combined_text = "\n\n".join(
        f"Source {i+1}: {source.get('title', 'Untitled')}\n"
        f"URL: {source.get('url', 'No URL')}\n"
        f"Content: {source.get('content', 'No content')[:2000]}..."
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
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1500,
    )
    return response.choices[0].message.content.strip()

def search_with_retry(query: str) -> List[Dict[str, Any]]:
    for attempt in range(MAX_RETRIES):
        try:
            response = tavily_client.search(
                query=query,
                search_depth="advanced",
                include_raw_content=True,
                max_results=7
            )
            return response.get("results", [])
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(DELAY_BETWEEN_REQUESTS)
            else:
                raise
    return []

@app.post("/research", response_model=ResearchResponse, tags=["Market Research"])
async def perform_iterative_research(request: ResearchRequest):
    try:
        all_sources = []
        iterations = []
        current_summary = ""
        query = generate_initial_query(
            request.domain,
            request.company_name,
            request.metrics,
            request.custom_operator
        )
        for iteration in range(MAX_ITERATIONS):
            print(f"Starting iteration {iteration + 1} with query: {query}")
            try:
                sources = search_with_retry(query)
                all_sources.extend(sources)
                time.sleep(DELAY_BETWEEN_REQUESTS)
                unique_sources = []
                seen_urls = set()
                for source in all_sources:
                    if source['url'] not in seen_urls:
                        seen_urls.add(source['url'])
                        unique_sources.append(source)
                all_sources = unique_sources
                if not sources:
                    if iteration == 0:
                        raise HTTPException(status_code=404, detail="No relevant sources found")
                    break
                current_summary = summarize_results(
                    sources,
                    request.domain,
                    request.metrics
                )
                if iteration < MAX_ITERATIONS - 1:
                    gaps = identify_knowledge_gaps(request.domain, current_summary)
                    if gaps:
                        query = generate_refinement_query(
                            request.domain,
                            current_summary,
                            gaps
                        )
                iterations.append({
                    "iteration": iteration + 1,
                    "query": query,
                    "sources_found": len(sources),
                    "summary": current_summary,
                    "knowledge_gaps": gaps if iteration < MAX_ITERATIONS - 1 else []
                })
            except Exception as e:
                print(f"Iteration {iteration + 1} failed: {str(e)}")
                if iteration == 0:
                    raise HTTPException(status_code=500, detail=f"Initial research failed: {str(e)}")
                break
        final_analysis = summarize_results(
            all_sources,
            request.domain,
            request.metrics
        )
        return {
            "final_analysis": final_analysis,
            "iterations": iterations,
            "all_sources": all_sources
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Research failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
