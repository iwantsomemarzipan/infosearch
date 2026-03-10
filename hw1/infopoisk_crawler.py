from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time
import random
import pandas as pd


class Crawler:
    def __init__(
        self,
        base_url = "https://randewoo.ru",
    ):
        self.base_url = base_url

        # настройка selenium
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_argument('User-Agent=Mozilla/5.0')

        self.driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=options
            )
        
    def _make_absolute_url(self, url):
        """Преобразует относительный URL в абсолютный"""
        if url.startswith('http'):
            return url
        return urljoin(self.base_url, url.strip())
    
    def get_product_urls_from_catalog(
        self,
        catalog_url,
        max_pages = 1
    ):
        """Собирает ссылки на товары со страницы каталога"""
        product_urls = []
        for page in range(1, max_pages + 1):
            page_url = f"{catalog_url}?page={page}"

            try:
                self.driver.get(page_url)
                time.sleep(3)

                items = self.driver.find_elements(
                    By.CSS_SELECTOR,
                    'li.products__item[data-role="catalog-products/item"]'
                    )
                
                for item in items:
                    relative_url = item.get_attribute('data-url')
                    if relative_url:
                        absolute_url = self._make_absolute_url(relative_url)
                        product_urls.append(absolute_url)

            except Exception as e:
                print(f"{page}: {e}")
                continue
        
        return product_urls

    def scrape_reviews(self, product_urls):
        """Собирает отзывы на товары"""
        all_reviews = []
        for url in product_urls:
            try:
                self.driver.get(url)
                time.sleep(random.uniform(1, 3))
                soup = BeautifulSoup(self.driver.page_source, 'html.parser')
                review_spans = soup.select('article.comment blockquote.comment__quote p span')

                for span in review_spans:
                    text = span.get_text(strip=True)
                    clean_text = ' '.join(text.split())
                    all_reviews.append(clean_text)

            except Exception as e:
                print(f"{url}: {e}")
                continue

        return all_reviews
    
    @staticmethod
    def save_to_csv(reviews, csv_path):
        """Сохраняет список отзывов в CSV-файл"""
        ids = list(range(len(reviews)))
        data = {'id': ids, 'review_text': reviews}
        df = pd.DataFrame(data)
        df.set_index('id', inplace=True)
        df.to_csv(csv_path)

    def close(self):
        """Закрывает браузер"""
        if self.driver:
            self.driver.quit()


crawler = Crawler()
urls = crawler.get_product_urls_from_catalog(
    "https://randewoo.ru/category/nishevaya-parfyumeriya",
    max_pages=3
    )

reviews = crawler.scrape_reviews(urls)
crawler.save_to_csv(reviews, 'reviews.csv')
crawler.close()

print(len(reviews))
