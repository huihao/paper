# LLM Paper Scraper - LLM论文抓取工具

A Python tool for scraping and fetching recent LLM (Large Language Model) research papers from arXiv.

## Features

- 🔍 Search for LLM-related papers on arXiv
- 📅 Fetch recent papers by date
- 🏷️ Filter by arXiv categories (cs.CL, cs.AI, cs.LG, etc.)
- 📄 Get paper metadata including title, authors, summary, and PDF links
- 🔧 Customizable search queries and parameters

## Installation

1. Clone the repository:
```bash
git clone https://github.com/huihao/paper.git
cd paper
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the default scraper to fetch recent LLM papers:

```bash
python paper_scraper.py
```

### Python API

You can also use the scraper as a library in your Python code:

```python
from paper_scraper import PaperScraper

# Initialize the scraper
scraper = PaperScraper()

# Search for recent LLM papers
papers = scraper.search_recent_llm_papers(max_results=10)

# Display papers
for paper in papers:
    print(f"Title: {paper['title']}")
    print(f"Authors: {', '.join(paper['authors'])}")
    print(f"PDF: {paper['pdf_url']}")
    print()

# Search by custom query
papers = scraper.search_papers(
    query="GPT OR ChatGPT",
    max_results=5,
    sort_by="submittedDate"
)

# Search by arXiv category
cl_papers = scraper.search_by_category(
    category="cs.CL",  # Computation and Language
    max_results=10
)
```

### Available Methods

#### `search_papers(query, max_results, sort_by, sort_order)`
Search for papers with a custom query.

**Parameters:**
- `query` (str): Search query string
- `max_results` (int): Maximum number of results (default: 10)
- `sort_by` (str): Sort criteria - "submittedDate", "lastUpdatedDate", or "relevance"
- `sort_order` (str): Sort order - "ascending" or "descending"

#### `search_recent_llm_papers(days, max_results)`
Search for recent LLM papers.

**Parameters:**
- `days` (int): Number of days to look back (default: 7)
- `max_results` (int): Maximum number of results (default: 20)

#### `search_by_category(category, max_results)`
Search papers by arXiv category.

**Parameters:**
- `category` (str): arXiv category code (e.g., "cs.CL", "cs.AI")
- `max_results` (int): Maximum number of results (default: 10)

### Common arXiv Categories

- `cs.CL` - Computation and Language (NLP)
- `cs.AI` - Artificial Intelligence
- `cs.LG` - Machine Learning
- `cs.CV` - Computer Vision
- `stat.ML` - Machine Learning (Statistics)

## Paper Data Structure

Each paper is returned as a dictionary with the following fields:

```python
{
    'title': str,           # Paper title
    'authors': List[str],   # List of author names
    'summary': str,         # Paper abstract/summary
    'published': str,       # Publication date
    'updated': str,         # Last update date
    'arxiv_id': str,        # arXiv identifier
    'pdf_url': str,         # Direct PDF download link
    'categories': List[str] # arXiv categories
}
```

## Configuration

You can customize search parameters in `config.yaml`:

- Modify default search keywords
- Add or remove arXiv categories
- Adjust result limits
- Configure time ranges

## Requirements

- Python 3.7+
- requests
- feedparser
- python-dateutil

## Examples

### Example 1: Find papers about transformers

```python
from paper_scraper import PaperScraper

scraper = PaperScraper()
papers = scraper.search_papers(query="transformer attention mechanism", max_results=5)

for paper in papers:
    print(paper['title'])
```

### Example 2: Get latest NLP papers

```python
from paper_scraper import PaperScraper

scraper = PaperScraper()
papers = scraper.search_by_category(category="cs.CL", max_results=10)

for paper in papers:
    print(f"{paper['title']} - {paper['published']}")
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.