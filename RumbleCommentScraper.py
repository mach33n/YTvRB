import time
import random

from selenium import webdriver
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

class RumbleCommentScraper:
    def __init__(self, username: str, password: str) -> None:
        #self.driver = webdriver.Chrome()
        self.driver = webdriver.Chrome(ChromeDriverManager().install())
        self.driver.implicitly_wait(0.5)

        self.username = username
        self.password = password

    def scrape_comments_from_url(self, url: str, num_comments: int):
        self.driver.get(url)

        # Log in to Rumble 
        self.driver.find_element(By.XPATH, '//button[text()="Sign In"]').click()
        self.driver.find_element(By.ID, 'login-username').send_keys(self.username)
        self.driver.find_element(By.ID, 'login-password').send_keys(self.password)
        self.driver.find_element(By.XPATH, '//button[text()="Sign in"]').click()

        # Wait briefly for page to load then scrape comment text and randomly sample if
        time.sleep(2)
        comments = self.driver.find_elements(By.CLASS_NAME, 'comment-text')
        comments = random.sample(comments, num_comments) if num_comments <= len(comments) else comments
        return [x.get_attribute("innerHTML") for x in comments]
