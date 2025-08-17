"""
SEC 10-K Analyzer using Anthropic API
This script fetches 10-K statements from SEC EDGAR and analyzes them using Anthropic's API.
"""

import os
from typing import Optional
from anthropic import Anthropic

# Install required packages:
# pip install edgartools anthropic

def get_10k_text(ticker: str, year: int) -> tuple[str, str]:
    """
    Fetch 10-K text for a given ticker and year.
    Returns a tuple of (full_text, truncated_text_for_api)
    """
    from edgar import Company, set_identity
    
    # Set your identity (required by SEC)
    set_identity("your.email@example.com")  # Replace with your actual email
    
    try:
        # Get the company
        company = Company(ticker)
        
        # Get 10-K filings
        filings = company.get_filings(form="10-K")
        
        # Find the filing for the specified year
        target_filing = None
        for filing in filings:
            filing_year = filing.filing_date.year
            # 10-K filed in early 2024 would be for fiscal year 2023
            if filing_year == year or filing_year == year + 1:
                target_filing = filing
                break
        
        if not target_filing:
            raise ValueError(f"No 10-K found for {ticker} for year {year}")
        
        print(f"Found 10-K: {target_filing.filing_date} - {target_filing.company_name}")
        print(f"Downloading and extracting text...")
        
        # Get the full text
        full_text = target_filing.text()
        
        # For API efficiency, we'll truncate to first 100,000 characters
        # (approximately 25,000 tokens, well within Claude's context window)
        # You can adjust this based on your needs
        max_chars = 100000
        
        if len(full_text) > max_chars:
            truncated_text = full_text[:max_chars]
            print(f"Note: 10-K text truncated from {len(full_text):,} to {max_chars:,} characters")
        else:
            truncated_text = full_text
            print(f"10-K text length: {len(full_text):,} characters")
        
        return full_text, truncated_text
        
    except Exception as e:
        raise Exception(f"Error fetching 10-K: {str(e)}")


def extract_specific_section(ticker: str, year: int, section: str) -> str:
    """
    Extract a specific section from the 10-K (e.g., 'Item 1A' for Risk Factors)
    This is more efficient for targeted questions about specific sections.
    """
    from edgar import Company, set_identity
    
    set_identity("your.email@example.com")  # Replace with your actual email
    
    try:
        company = Company(ticker)
        filings = company.get_filings(form="10-K")
        
        # Find the filing for the specified year
        target_filing = None
        for filing in filings:
            filing_year = filing.filing_date.year
            if filing_year == year or filing_year == year + 1:
                target_filing = filing
                break
        
        if not target_filing:
            raise ValueError(f"No 10-K found for {ticker} for year {year}")
        
        # Get the filing document
        doc = target_filing.html()
        
        # Note: Section extraction would require additional parsing
        # For now, we'll return the full text with a note
        return target_filing.text()
        
    except Exception as e:
        raise Exception(f"Error extracting section: {str(e)}")


def analyze_10k_with_anthropic(
    ticker: str,
    year: int,
    question: str,
    api_key: str,
    model: str = "claude-3-5-sonnet-20241022",
    use_full_text: bool = False
) -> str:
    """
    Analyze a 10-K filing using Anthropic's API.
    
    Args:
        ticker: Stock ticker symbol
        year: Fiscal year for the 10-K
        question: Question to ask about the 10-K
        api_key: Anthropic API key
        model: Anthropic model to use
        use_full_text: Whether to use full text (may exceed context for very large filings)
    
    Returns:
        Analysis from Anthropic
    """
    
    # Initialize Anthropic client
    client = Anthropic(api_key=api_key)
    
    print(f"\nFetching 10-K for {ticker} (fiscal year {year})...")
    
    try:
        # Get the 10-K text
        full_text, truncated_text = get_10k_text(ticker, year)
        
        # Use appropriate text based on preference
        text_to_analyze = full_text if use_full_text else truncated_text
        
        # Prepare the prompt
        prompt = f"""You are analyzing the 10-K filing for {ticker} (fiscal year {year}).

Based on the following 10-K content, please answer this question:
{question}

Please provide a detailed, specific answer based only on the information in the 10-K filing. 
If the information to answer the question is not in the provided text, please state that clearly.

10-K CONTENT:
{text_to_analyze}
"""
        
        print(f"\nSending to Anthropic {model}...")
        print(f"Question: {question}")
        print("\nGenerating analysis...")
        
        # Send to Anthropic
        response = client.messages.create(
            model=model,
            max_tokens=4000,
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        
        return response.content[0].text
        
    except Exception as e:
        return f"Error: {str(e)}"


def main():
    """
    Main function to run the 10-K analyzer
    """
    print("=" * 60)
    print("SEC 10-K ANALYZER WITH ANTHROPIC API")
    print("=" * 60)
    
    # Get user inputs
    print("\nPlease provide the following information:")
    
    ticker = input("Company ticker symbol (e.g., AAPL): ").strip().upper()
    
    year = input("Fiscal year (e.g., 2023): ").strip()
    try:
        year = int(year)
    except ValueError:
        print("Invalid year. Please enter a valid year.")
        return
    
    question = input("Your question about the 10-K: ").strip()
    
    api_key = input("Your Anthropic API key: ").strip()
    
    print("\nAvailable Anthropic models:")
    print("1. claude-3-5-sonnet-20241022 (recommended - best balance)")
    print("2. claude-3-5-haiku-20241022 (faster, cheaper)")
    print("3. claude-3-opus-20240229 (most capable)")
    
    model_choice = input("Choose model (1-3, default is 1): ").strip()
    
    model_map = {
        "1": "claude-3-5-sonnet-20241022",
        "2": "claude-3-5-haiku-20241022",
        "3": "claude-3-opus-20240229",
        "": "claude-3-5-sonnet-20241022"  # default
    }
    
    model = model_map.get(model_choice, "claude-3-5-sonnet-20241022")
    
    # Analyze the 10-K
    print("\n" + "=" * 60)
    result = analyze_10k_with_anthropic(
        ticker=ticker,
        year=year,
        question=question,
        api_key=api_key,
        model=model
    )
    
    # Display results
    print("\n" + "=" * 60)
    print("ANALYSIS RESULT:")
    print("=" * 60)
    print(result)
    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Example usage (commented out - uncomment to use programmatically)
    """
    # Example 1: Analyze Apple's China exposure
    result = analyze_10k_with_anthropic(
        ticker="AAPL",
        year=2023,
        question="How exposed is this firm to China and how? Provide specific revenue figures and operational dependencies.",
        api_key="your-api-key-here",
        model="claude-3-5-sonnet-20241022"
    )
    print(result)
    
    # Example 2: Analyze Tesla's risk factors
    result = analyze_10k_with_anthropic(
        ticker="TSLA",
        year=2023,
        question="What are the top 5 risk factors mentioned in the 10-K?",
        api_key="your-api-key-here",
        model="claude-3-5-sonnet-20241022"
    )
    print(result)
    """
    
    # Run interactive mode
    main()