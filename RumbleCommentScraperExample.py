import RumbleCommentScraper as RCS

scraper = RCS.RumbleCommentScraper(username='CS6474', password='georgiatech')
comments = scraper.scrape_comments_from_url('https://rumble.com/v2gcxvy--live-daily-show-louder-with-crowder.html', 50)

for comment in comments:
    print(comment)
