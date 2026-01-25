import asyncio
import random
from collections import deque
from googleapiclient.discovery import build
from config import YOUTUBE_API_KEY, LIVE_ID
from src.tools.base_tool import BaseTool


class YoutubeCommentManager:
    def __init__(self, api_key: str, live_id: str):
        """
        Initialize YouTube Comment Manager.
        
        Args:
            api_key: YouTube Data API v3 key
            live_id: YouTube Live video ID
        """
        self.api_key = api_key
        self.live_id = live_id
        self.comment_queue = deque(maxlen=10)  # Maximum 10 comments
        self.youtube = build('youtube', 'v3', developerKey=api_key)
        self.next_page_token = None
        self.live_chat_id = None
        self.running = False

    async def start_polling(self):
        """Start the comment polling loop (runs every 10 seconds)."""
        self.running = True
        print("-> YouTube comment polling started.")
        
        # Get live chat ID first
        try:
            video_response = await asyncio.to_thread(
                self.youtube.videos().list(
                    part='liveStreamingDetails',
                    id=self.live_id
                ).execute
            )
            
            if video_response['items']:
                self.live_chat_id = video_response['items'][0]['liveStreamingDetails'].get('activeLiveChatId')
                if not self.live_chat_id:
                    print("   [YouTube] No active live chat found.")
                    self.running = False
                    return
                print(f"   [YouTube] Live chat ID: {self.live_chat_id}")
            else:
                print("   [YouTube] Video not found.")
                self.running = False
                return
        except Exception as e:
            error_msg = str(e)
            if "API key expired" in error_msg or "badRequest" in error_msg:
                print("   [YouTube Error] API key has expired. Please renew your YouTube API key in .env file.")
            elif "quota" in error_msg.lower():
                print("   [YouTube Error] API quota exceeded. Please wait or check your quota limits.")
            else:
                print(f"   [YouTube Init Error]: {e}")
            self.running = False
            return
        
        while self.running:
            try:
                await self._fetch_comments()
            except Exception as e:
                print(f"   [YouTube Comment Error]: {e}")
            
            await asyncio.sleep(10)  # Poll every 10 seconds

    async def stop_polling(self):
        """Stop the comment polling loop."""
        self.running = False
        print("-> YouTube comment polling stopped.")

    async def _fetch_comments(self):
        """Fetch new comments from YouTube Live chat."""
        try:
            # Fetch live chat messages
            chat_response = await asyncio.to_thread(
                self.youtube.liveChatMessages().list(
                    liveChatId=self.live_chat_id,
                    part='snippet',
                    pageToken=self.next_page_token
                ).execute
            )

            self.next_page_token = chat_response.get('nextPageToken')

            # Add new comments to queue (without username)
            for item in chat_response.get('items', []):
                message = item['snippet']['displayMessage']
                self.comment_queue.append(message)
                print(f"   [YouTube Comment] Added: {message}")

        except Exception as e:
            print(f"   [YouTube Fetch Error]: {e}")

    async def get_random_comment(self) -> str:
        """
        Get a random comment from the queue and remove it.
        
        Returns:
            Comment text, or None if queue is empty
        """
        if not self.comment_queue:
            return None
        
        # Get random comment and remove it from queue
        comment = random.choice(self.comment_queue)
        self.comment_queue.remove(comment)
        return comment
    
    def has_comments(self) -> bool:
        """Check if there are comments in the queue."""
        return len(self.comment_queue) > 0
    
    def get_comment_count(self) -> int:
        """Get the number of comments in the queue."""
        return len(self.comment_queue)


class YoutubeCommentTool(BaseTool):
    def __init__(self, comment_manager=None):
        self.comment_manager = comment_manager

    @property
    def name(self) -> str:
        return "youtube_comment"

    @property
    def description(self) -> str:
        return "Get a random comment from YouTube Live chat"

    async def execute(self, **kwargs) -> str:
        """
        Get a random comment from the YouTube comment queue.
        Returns the comment text, or a message if no comments are available.
        """
        if not self.comment_manager:
            return "YouTubeコメント機能が無効です。"
        
        comment = await self.comment_manager.get_random_comment()
        if comment:
            return f"視聴者のコメント: {comment}"
        else:
            return "現在、新しいコメントはありません。"
