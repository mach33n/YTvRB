import requests
import pickle
import os

example_ID_list = [("Topic", "Title", "_VB39Jo8mAQ")]

class YoutubeCommentScraper:
    def __init__(self, commentCount: int = 1000, API_Key: str = "AIzaSyDUQyRdH2MOj-JG0bjLrydbNVC7-EjzCqo"):
        self.part = "snippet%2Creplies"
        self.key = API_Key
        self.commentCount = commentCount

    def scrape_comments_from_list(self, IDList: list):
        comments = []
        count = 0
        pageToken = ""
        for topic, title, idstr in IDList:
            while count < self.commentCount:
                out = requests.get(f"https://youtube.googleapis.com/youtube/v3/commentThreads?part={self.part}&videoId={idstr}&key={self.key}&maxResults=100&pageToken={pageToken}")
                resp = out.json()
                count += int(resp["pageInfo"]["totalResults"])
                comments += map(lambda x: (x["id"], x["snippet"]["topLevelComment"]["snippet"]["textOriginal"], x["snippet"]["totalReplyCount"]), resp["items"])
                pageToken = resp["nextPageToken"]
            pageToken = ""
        
            # Pickle comments into file
            os.makedirs(os.path.dirname(f"YT/{topic}/{title}.pkl"), exist_ok=True)
            with open(f"YT/{topic}/{title}.pkl", 'wb') as titleFile:
                pickle.dump(comments, titleFile)
