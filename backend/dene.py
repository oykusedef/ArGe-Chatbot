import requests

API_KEY = "PPIqVOjLOi5W7Jsk3cXjUHmMJwRFDA9DLXlYhejX"
url = f"https://newsapi.org/v2/everything?q=borsa&language=tr&pageSize=5&apiKey={API_KEY}"

response = requests.get(url)
data = response.json()

for i, article in enumerate(data["articles"], 1):
    print(f"{i}. {article['title']}\n{article['url']}\n")
