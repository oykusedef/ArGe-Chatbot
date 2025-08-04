import os
from newsapi import NewsApiClient
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv("api.env")

# Get API key
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "PPIqVOjLOi5W7Jsk3cXjUHmMJwRFDA9DLXlYhejX")
print(f"Using News API Key: {NEWS_API_KEY[:10]}...")

# Initialize News API client
newsapi = NewsApiClient(api_key=NEWS_API_KEY)

def test_news_api():
    """Test News API functionality"""
    print("Testing News API...")
    
    # Test 1: Basic search for AKBNK
    print("\n1. Testing AKBNK search:")
    try:
        news = newsapi.get_everything(
            q='AKBNK',
            language='tr',
            sort_by='publishedAt',
            from_param=(datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
            page_size=5
        )
        print(f"Status: {news['status']}")
        print(f"Total Results: {news.get('totalResults', 0)}")
        print(f"Articles found: {len(news.get('articles', []))}")
        
        if news.get('articles'):
            for i, article in enumerate(news['articles'][:3], 1):
                print(f"  {i}. {article.get('title', 'No title')}")
                print(f"     Source: {article.get('source', {}).get('name', 'Unknown')}")
                print(f"     Date: {article.get('publishedAt', 'Unknown')}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 2: Search for Akbank (company name)
    print("\n2. Testing Akbank search:")
    try:
        news = newsapi.get_everything(
            q='Akbank',
            language='tr',
            sort_by='publishedAt',
            from_param=(datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
            page_size=5
        )
        print(f"Status: {news['status']}")
        print(f"Total Results: {news.get('totalResults', 0)}")
        print(f"Articles found: {len(news.get('articles', []))}")
        
        if news.get('articles'):
            for i, article in enumerate(news['articles'][:3], 1):
                print(f"  {i}. {article.get('title', 'No title')}")
                print(f"     Source: {article.get('source', {}).get('name', 'Unknown')}")
                print(f"     Date: {article.get('publishedAt', 'Unknown')}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 3: Search for ASELS
    print("\n3. Testing ASELS search:")
    try:
        news = newsapi.get_everything(
            q='ASELS',
            language='tr',
            sort_by='publishedAt',
            from_param=(datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
            page_size=5
        )
        print(f"Status: {news['status']}")
        print(f"Total Results: {news.get('totalResults', 0)}")
        print(f"Articles found: {len(news.get('articles', []))}")
        
        if news.get('articles'):
            for i, article in enumerate(news['articles'][:3], 1):
                print(f"  {i}. {article.get('title', 'No title')}")
                print(f"     Source: {article.get('source', {}).get('name', 'Unknown')}")
                print(f"     Date: {article.get('publishedAt', 'Unknown')}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 4: Search for Aselsan (company name)
    print("\n4. Testing Aselsan search:")
    try:
        news = newsapi.get_everything(
            q='Aselsan',
            language='tr',
            sort_by='publishedAt',
            from_param=(datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
            page_size=5
        )
        print(f"Status: {news['status']}")
        print(f"Total Results: {news.get('totalResults', 0)}")
        print(f"Articles found: {len(news.get('articles', []))}")
        
        if news.get('articles'):
            for i, article in enumerate(news['articles'][:3], 1):
                print(f"  {i}. {article.get('title', 'No title')}")
                print(f"     Source: {article.get('source', {}).get('name', 'Unknown')}")
                print(f"     Date: {article.get('publishedAt', 'Unknown')}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 5: English search as fallback
    print("\n5. Testing English search for AKBNK:")
    try:
        news = newsapi.get_everything(
            q='AKBNK',
            language='en',
            sort_by='publishedAt',
            from_param=(datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
            page_size=5
        )
        print(f"Status: {news['status']}")
        print(f"Total Results: {news.get('totalResults', 0)}")
        print(f"Articles found: {len(news.get('articles', []))}")
        
        if news.get('articles'):
            for i, article in enumerate(news['articles'][:3], 1):
                print(f"  {i}. {article.get('title', 'No title')}")
                print(f"     Source: {article.get('source', {}).get('name', 'Unknown')}")
                print(f"     Date: {article.get('publishedAt', 'Unknown')}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_news_api() 