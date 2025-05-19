# import requests

# url = "http://localhost:8000/research"
# data = {
#     "domain": "electric vehicle charging infrastructure",
#     "company_name": "ChargePoint",
#     "metrics": ["market share", "growth rate"],
#     "custom_operator": "SWOT analysis"
# }

# response = requests.post(url, json=data)
# print(response.json()["detailed_analysis"])
import requests
import json
import time
from typing import Optional, List

# Configuration
API_URL = "http://localhost:8000/research"  # Update if hosted elsewhere
DEFAULT_TIMEOUT = 300  # 5 minutes for long-running research

def conduct_market_research(
    domain: str,
    company_name: Optional[str] = None,
    metrics: Optional[List[str]] = None,
    custom_operator: Optional[str] = None,
    timeout: int = DEFAULT_TIMEOUT
) -> dict:
    """
    Perform iterative market research via the API.
    
    Args:
        domain: Industry domain to research (required)
        company_name: Specific company to analyze (optional)
        metrics: List of metrics to focus on (optional)
        custom_operator: Analysis method like "SWOT" (optional)
        timeout: Request timeout in seconds
        
    Returns:
        Dictionary with research results
        
    Raises:
        requests.exceptions.RequestException: On API communication failures
        ValueError: On invalid responses
    """
    payload = {
        "domain": domain,
        "company_name": company_name,
        "metrics": metrics or [],
        "custom_operator": custom_operator
    }

    try:
        start_time = time.time()
        response = requests.post(
            API_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=timeout
        )
        response.raise_for_status()
        
        data = response.json()
        processing_time = time.time() - start_time
        
        # Enhance response with timing info
        data["metadata"] = {
            "processing_time_sec": round(processing_time, 2),
            "api_status": "success"
        }
        
        return data
        
    except requests.exceptions.HTTPError as e:
        error_detail = str(e)
        if e.response is not None:
            try:
                error_detail = e.response.json().get("detail", str(e))
            except ValueError:
                error_detail = e.response.text
        raise requests.exceptions.RequestException(f"API Error: {error_detail}")
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON response from API")

def print_research_summary(results: dict):
    """Print formatted research summary to console"""
    print("\n" + "="*50)
    print("MARKET RESEARCH REPORT SUMMARY")
    print("="*50)
    
    # Basic info
    first_query = results["iterations"][0]["query"]
    domain = first_query.split("about:")[-1].split(" in the")[0].strip()
    print(f"\nDomain: {domain}")
    
    if results["iterations"][0].get("company_name"):
        print(f"Company: {results['iterations'][0]['company_name']}")
    
    # Timing info
    if "metadata" in results:
        print(f"\nProcessing Time: {results['metadata']['processing_time_sec']} seconds")
    
    # Final analysis
    print("\n★ Key Findings ★")
    print("-"*50)
    print(results["final_analysis"][:2000] + "...")  # Preview
    
    # Research process
    print("\n\n★ Research Process ★")
    print("-"*50)
    for i, iteration in enumerate(results["iterations"], 1):
        print(f"\nIteration {i}:")
        print(f"Query: {iteration['query'][:150]}...")
        print(f"Sources Found: {iteration['sources_found']}")
        
        if iteration.get("knowledge_gaps"):
            print("\nIdentified Gaps:")
            for gap in iteration["knowledge_gaps"]:
                print(f"  - {gap[:100]}...")
    
    print(f"\nTotal Sources Analyzed: {len(results['all_sources'])}")
    print("\n" + "="*50)

def save_full_report(results: dict, filename: Optional[str] = None):
    """Save complete research results to JSON file"""
    if not filename:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"market_research_{timestamp}.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nFull report saved to: {filename}")

if __name__ == "__main__":
    # Get research parameters from the user
    domain = input("Enter the domain to research: ")
    company_name = input("Enter the company name (optional): ") or None
    metrics_input = input("Enter metrics to analyze, separated by commas (optional): ") or None
    metrics = [m.strip() for m in metrics_input.split(",")] if metrics_input else None
    custom_operator = input("Enter a custom analysis operator (optional): ") or None

    try:
        print(f"\nStarting research on: {domain}")
        if company_name:
            print(f"Focusing on company: {company_name}")

        results = conduct_market_research(
            domain=domain,
            company_name=company_name,
            metrics=metrics,
            custom_operator=custom_operator
        )
        print_research_summary(results)
        save_full_report(results)

    except Exception as e:
        print("\nResearch failed:")
        print(f"Error: {str(e)}")
