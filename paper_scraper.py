#!/usr/bin/env python3
"""
LLM Paper Scraper - 论文抓取工具
Fetches recent LLM research papers from arXiv
"""

import feedparser
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import time


class PaperScraper:
    """Scraper for fetching LLM research papers from arXiv"""
    
    def __init__(self):
        self.base_url = "http://export.arxiv.org/api/query"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (compatible; LLM-Paper-Scraper/1.0)'
        }
    
    def search_papers(
        self, 
        query: str = "LLM OR \"large language model\"",
        max_results: int = 10,
        sort_by: str = "submittedDate",
        sort_order: str = "descending"
    ) -> List[Dict]:
        """
        Search for papers on arXiv
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            sort_by: Sort criteria (submittedDate, lastUpdatedDate, relevance)
            sort_order: Sort order (ascending, descending)
            
        Returns:
            List of paper dictionaries containing title, authors, summary, etc.
        """
        params = {
            'search_query': f'all:{query}',
            'start': 0,
            'max_results': max_results,
            'sortBy': sort_by,
            'sortOrder': sort_order
        }
        
        try:
            response = requests.get(self.base_url, params=params, headers=self.headers)
            response.raise_for_status()
            
            feed = feedparser.parse(response.content)
            
            papers = []
            for entry in feed.entries:
                paper = {
                    'title': entry.title,
                    'authors': [author.name for author in entry.authors],
                    'summary': entry.summary,
                    'published': entry.published,
                    'updated': entry.updated,
                    'arxiv_id': entry.id.split('/abs/')[-1],
                    'pdf_url': entry.id.replace('/abs/', '/pdf/') + '.pdf',
                    'categories': [tag.term for tag in entry.tags] if hasattr(entry, 'tags') else []
                }
                papers.append(paper)
            
            return papers
            
        except Exception as e:
            print(f"Error fetching papers: {e}")
            return []
    
    def search_recent_llm_papers(self, days: int = 7, max_results: int = 20) -> List[Dict]:
        """
        Search for recent LLM papers
        
        Args:
            days: Number of days to look back
            max_results: Maximum number of results
            
        Returns:
            List of recent LLM papers
        """
        queries = [
            "LLM",
            "large language model",
            "GPT",
            "transformer",
            "BERT",
            "natural language processing"
        ]
        
        # Combine queries with OR
        combined_query = " OR ".join([f'"{q}"' for q in queries])
        
        return self.search_papers(
            query=combined_query,
            max_results=max_results,
            sort_by="submittedDate"
        )
    
    def search_by_category(self, category: str = "cs.CL", max_results: int = 10) -> List[Dict]:
        """
        Search papers by arXiv category
        
        Args:
            category: arXiv category (e.g., cs.CL for Computation and Language)
            max_results: Maximum number of results
            
        Returns:
            List of papers in the specified category
        """
        params = {
            'search_query': f'cat:{category}',
            'start': 0,
            'max_results': max_results,
            'sortBy': 'submittedDate',
            'sortOrder': 'descending'
        }
        
        try:
            response = requests.get(self.base_url, params=params, headers=self.headers)
            response.raise_for_status()
            
            feed = feedparser.parse(response.content)
            
            papers = []
            for entry in feed.entries:
                paper = {
                    'title': entry.title,
                    'authors': [author.name for author in entry.authors],
                    'summary': entry.summary,
                    'published': entry.published,
                    'updated': entry.updated,
                    'arxiv_id': entry.id.split('/abs/')[-1],
                    'pdf_url': entry.id.replace('/abs/', '/pdf/') + '.pdf',
                    'categories': [tag.term for tag in entry.tags] if hasattr(entry, 'tags') else []
                }
                papers.append(paper)
            
            return papers
            
        except Exception as e:
            print(f"Error fetching papers by category: {e}")
            return []
    
    def format_paper_info(self, paper: Dict) -> str:
        """Format paper information for display"""
        authors_str = ", ".join(paper['authors'][:3])
        if len(paper['authors']) > 3:
            authors_str += f" et al. ({len(paper['authors'])} authors)"
        
        info = f"""
Title: {paper['title']}
Authors: {authors_str}
Published: {paper['published']}
arXiv ID: {paper['arxiv_id']}
PDF: {paper['pdf_url']}
Categories: {', '.join(paper['categories'])}

Summary:
{paper['summary'][:300]}...

{'='*80}
"""
        return info


def main():
    """Main function to demonstrate usage"""
    scraper = PaperScraper()
    
    print("Fetching recent LLM papers from arXiv...")
    print("="*80)
    
    # Search for recent LLM papers
    papers = scraper.search_recent_llm_papers(max_results=5)
    
    if papers:
        print(f"\nFound {len(papers)} papers:\n")
        for i, paper in enumerate(papers, 1):
            print(f"\n--- Paper {i} ---")
            print(scraper.format_paper_info(paper))
    else:
        print("No papers found.")
    
    # Also search by category
    print("\n\nFetching papers from cs.CL (Computation and Language) category...")
    print("="*80)
    cl_papers = scraper.search_by_category(category="cs.CL", max_results=3)
    
    if cl_papers:
        print(f"\nFound {len(cl_papers)} papers in cs.CL:\n")
        for i, paper in enumerate(cl_papers, 1):
            print(f"\n--- Paper {i} ---")
            print(scraper.format_paper_info(paper))


if __name__ == "__main__":
    main()
