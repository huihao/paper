#!/usr/bin/env python3
"""
Unit tests for the LLM Paper Scraper
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from paper_scraper import PaperScraper


class TestPaperScraper(unittest.TestCase):
    """Test cases for PaperScraper class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.scraper = PaperScraper()
        
        # Mock paper data
        self.mock_entry = Mock()
        self.mock_entry.title = "Test Paper on Large Language Models"
        self.mock_entry.summary = "This is a test summary for an LLM paper."
        self.mock_entry.published = "2024-01-01T00:00:00Z"
        self.mock_entry.updated = "2024-01-02T00:00:00Z"
        self.mock_entry.id = "http://arxiv.org/abs/2401.00001"
        
        mock_author1 = Mock()
        mock_author1.name = "John Doe"
        mock_author2 = Mock()
        mock_author2.name = "Jane Smith"
        self.mock_entry.authors = [mock_author1, mock_author2]
        
        mock_tag1 = Mock()
        mock_tag1.term = "cs.CL"
        mock_tag2 = Mock()
        mock_tag2.term = "cs.AI"
        self.mock_entry.tags = [mock_tag1, mock_tag2]
    
    def test_initialization(self):
        """Test scraper initialization"""
        self.assertEqual(self.scraper.base_url, "http://export.arxiv.org/api/query")
        self.assertIn('User-Agent', self.scraper.headers)
    
    @patch('paper_scraper.requests.get')
    @patch('paper_scraper.feedparser.parse')
    def test_search_papers(self, mock_parse, mock_get):
        """Test search_papers method"""
        # Setup mocks
        mock_response = Mock()
        mock_response.content = b"mock content"
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        mock_feed = Mock()
        mock_feed.entries = [self.mock_entry]
        mock_parse.return_value = mock_feed
        
        # Execute
        papers = self.scraper.search_papers(query="LLM", max_results=1)
        
        # Verify
        self.assertEqual(len(papers), 1)
        self.assertEqual(papers[0]['title'], "Test Paper on Large Language Models")
        self.assertEqual(len(papers[0]['authors']), 2)
        self.assertIn("John Doe", papers[0]['authors'])
        self.assertEqual(papers[0]['arxiv_id'], "2401.00001")
        self.assertIn('cs.CL', papers[0]['categories'])
    
    @patch('paper_scraper.requests.get')
    @patch('paper_scraper.feedparser.parse')
    def test_search_by_category(self, mock_parse, mock_get):
        """Test search_by_category method"""
        # Setup mocks
        mock_response = Mock()
        mock_response.content = b"mock content"
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        mock_feed = Mock()
        mock_feed.entries = [self.mock_entry]
        mock_parse.return_value = mock_feed
        
        # Execute
        papers = self.scraper.search_by_category(category="cs.CL", max_results=1)
        
        # Verify
        self.assertEqual(len(papers), 1)
        self.assertEqual(papers[0]['title'], "Test Paper on Large Language Models")
        
        # Check that the correct API call was made
        mock_get.assert_called_once()
        call_args = mock_get.call_args
        self.assertIn('cat:cs.CL', str(call_args))
    
    @patch('paper_scraper.PaperScraper.search_papers')
    def test_search_recent_llm_papers(self, mock_search):
        """Test search_recent_llm_papers method"""
        # Setup mock
        mock_search.return_value = [{
            'title': 'Test LLM Paper',
            'authors': ['Test Author'],
            'summary': 'Test summary',
            'published': '2024-01-01',
            'updated': '2024-01-01',
            'arxiv_id': '2401.00001',
            'pdf_url': 'http://arxiv.org/pdf/2401.00001.pdf',
            'categories': ['cs.CL']
        }]
        
        # Execute
        papers = self.scraper.search_recent_llm_papers(max_results=1)
        
        # Verify
        self.assertEqual(len(papers), 1)
        mock_search.assert_called_once()
    
    def test_format_paper_info(self):
        """Test format_paper_info method"""
        paper = {
            'title': 'Test Paper',
            'authors': ['Author 1', 'Author 2', 'Author 3', 'Author 4'],
            'summary': 'A' * 400,  # Long summary
            'published': '2024-01-01',
            'arxiv_id': '2401.00001',
            'pdf_url': 'http://arxiv.org/pdf/2401.00001.pdf',
            'categories': ['cs.CL', 'cs.AI']
        }
        
        # Execute
        formatted = self.scraper.format_paper_info(paper)
        
        # Verify
        self.assertIn('Test Paper', formatted)
        self.assertIn('Author 1', formatted)
        self.assertIn('et al.', formatted)  # More than 3 authors
        self.assertIn('2401.00001', formatted)
        self.assertIn('cs.CL', formatted)
    
    @patch('paper_scraper.requests.get')
    def test_error_handling(self, mock_get):
        """Test error handling when API call fails"""
        # Setup mock to raise exception
        mock_get.side_effect = Exception("Network error")
        
        # Execute
        papers = self.scraper.search_papers(query="LLM")
        
        # Verify graceful handling
        self.assertEqual(papers, [])


class TestPaperStructure(unittest.TestCase):
    """Test paper data structure"""
    
    def test_paper_dict_structure(self):
        """Test that paper dictionary has expected fields"""
        expected_fields = [
            'title', 'authors', 'summary', 'published',
            'updated', 'arxiv_id', 'pdf_url', 'categories'
        ]
        
        # Create a sample paper
        paper = {
            'title': 'Test',
            'authors': ['Author'],
            'summary': 'Summary',
            'published': '2024-01-01',
            'updated': '2024-01-01',
            'arxiv_id': '2401.00001',
            'pdf_url': 'http://test.pdf',
            'categories': ['cs.CL']
        }
        
        for field in expected_fields:
            self.assertIn(field, paper)


if __name__ == '__main__':
    unittest.main()
