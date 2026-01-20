from googleapiclient.discovery import build

YOUTUBE_API_KEY = "AIzaSyATyvdqr8z0pVSTi3VOGLQ6BKyRrzL_7NI"
live_id = "QwglvIyCdKY"

youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
res = youtube.videos().list(part='liveStreamingDetails', id=live_id).execute()
chat_id = res['items'][0]['liveStreamingDetails']['activeLiveChatId']
res = youtube.liveChatMessages().list(liveChatId=chat_id, part='snippet', maxResults=200).execute()
print(res)