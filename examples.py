#!/usr/bin/env python3
"""
Example script demonstrating various uses of the LLM Paper Scraper
"""

from paper_scraper import PaperScraper


def example_1_basic_search():
    """Example 1: Basic search for LLM papers"""
    print("\n" + "="*80)
    print("Example 1: Basic Search for LLM Papers")
    print("="*80)
    
    scraper = PaperScraper()
    papers = scraper.search_recent_llm_papers(max_results=3)
    
    for i, paper in enumerate(papers, 1):
        print(f"\n{i}. {paper['title']}")
        print(f"   Authors: {', '.join(paper['authors'][:2])}")
        print(f"   Published: {paper['published']}")
        print(f"   arXiv: {paper['arxiv_id']}")


def example_2_category_search():
    """Example 2: Search by arXiv category"""
    print("\n" + "="*80)
    print("Example 2: Search by Category (cs.CL - Computation and Language)")
    print("="*80)
    
    scraper = PaperScraper()
    papers = scraper.search_by_category(category="cs.CL", max_results=3)
    
    for i, paper in enumerate(papers, 1):
        print(f"\n{i}. {paper['title']}")
        print(f"   Categories: {', '.join(paper['categories'][:3])}")


def example_3_custom_query():
    """Example 3: Custom query search"""
    print("\n" + "="*80)
    print("Example 3: Custom Query - GPT and ChatGPT papers")
    print("="*80)
    
    scraper = PaperScraper()
    papers = scraper.search_papers(
        query="GPT OR ChatGPT",
        max_results=3,
        sort_by="submittedDate"
    )
    
    for i, paper in enumerate(papers, 1):
        print(f"\n{i}. {paper['title']}")
        print(f"   PDF: {paper['pdf_url']}")


def example_4_detailed_info():
    """Example 4: Get detailed paper information"""
    print("\n" + "="*80)
    print("Example 4: Detailed Paper Information")
    print("="*80)
    
    scraper = PaperScraper()
    papers = scraper.search_papers(query="BERT", max_results=1)
    
    if papers:
        paper = papers[0]
        print(scraper.format_paper_info(paper))


def example_5_multiple_categories():
    """Example 5: Search across multiple categories"""
    print("\n" + "="*80)
    print("Example 5: Search Across Multiple Categories")
    print("="*80)
    
    scraper = PaperScraper()
    categories = ["cs.CL", "cs.AI", "cs.LG"]
    
    for category in categories:
        papers = scraper.search_by_category(category=category, max_results=2)
        print(f"\n{category}: Found {len(papers)} papers")
        if papers:
            print(f"  Latest: {papers[0]['title'][:60]}...")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("LLM Paper Scraper - Example Demonstrations")
    print("="*80)
    
    # Run all examples
    example_1_basic_search()
    example_2_category_search()
    example_3_custom_query()
    example_4_detailed_info()
    example_5_multiple_categories()
    
    print("\n" + "="*80)
    print("All examples completed!")
    print("="*80 + "\n")
