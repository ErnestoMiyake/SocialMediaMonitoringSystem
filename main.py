import os
import json
import sqlite3
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import asyncio
from pathlib import Path

# ==================== ВИПРАВЛЕННЯ ДЛЯ WINDOWS ====================
import sys
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
import uvicorn
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from cryptography.fernet import Fernet
import logging

# Scrapers
from telethon import TelegramClient
from playwright.sync_api import sync_playwright, Page as SyncPage
import requests
from bs4 import BeautifulSoup

# ML для аналізу тональності
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import re

# Налаштування логування
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== КОНФІГУРАЦІЯ ====================
DB_PATH = "db.db"
BOGON_PATH = "bogon.json"
SESSION_DIR = ".session"
KEY_FILE = "encryption.key"

if os.path.exists(KEY_FILE):
    with open(KEY_FILE, "r") as f:
        ENCRYPTION_KEY = f.read().strip()
else:
    ENCRYPTION_KEY = Fernet.generate_key().decode()
    with open(KEY_FILE, "w") as f:
        f.write(ENCRYPTION_KEY)

fernet = Fernet(ENCRYPTION_KEY.encode())

# Створення директорій
Path(SESSION_DIR).mkdir(exist_ok=True)

app = FastAPI()
security = HTTPBasic()
scheduler = AsyncIOScheduler()

# ==================== DATABASE ====================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Таблиця користувачів
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        role TEXT NOT NULL DEFAULT 'user',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    
    # Таблиця джерел
    c.execute('''CREATE TABLE IF NOT EXISTS sources (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        url TEXT UNIQUE NOT NULL,
        platform TEXT NOT NULL,
        title TEXT,
        active INTEGER DEFAULT 1,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    
    # Таблиця метрик
    c.execute('''CREATE TABLE IF NOT EXISTS metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source_id INTEGER NOT NULL,
        views INTEGER DEFAULT 0,
        likes INTEGER DEFAULT 0,
        comments_count INTEGER DEFAULT 0,
        reposts INTEGER DEFAULT 0,
        negative_count INTEGER DEFAULT 0,
        collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (source_id) REFERENCES sources (id)
    )''')
    
    # Таблиця коментарів
    c.execute('''CREATE TABLE IF NOT EXISTS comments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source_id INTEGER NOT NULL,
        text TEXT NOT NULL,
        sentiment TEXT NOT NULL,
        confidence REAL,
        author TEXT,
        collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (source_id) REFERENCES sources (id)
    )''')
    
    # Таблиця API ключів
    c.execute('''CREATE TABLE IF NOT EXISTS api_keys (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        platform TEXT NOT NULL,
        key_name TEXT NOT NULL,
        key_value TEXT NOT NULL,
        expires_at TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    
    # Таблиця логів
    c.execute('''CREATE TABLE IF NOT EXISTS logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user TEXT,
        action TEXT,
        details TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    
    # Створення адміна за замовчуванням
    admin_pass = hashlib.sha256("admin123".encode()).hexdigest()
    try:
        c.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
                 ("admin", admin_pass, "admin"))
    except sqlite3.IntegrityError:
        pass
    
    conn.commit()
    conn.close()

init_db()

# ==================== SENTIMENT ANALYZER ====================
class SentimentAnalyzer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.model = MultinomialNB()
        self.trained = False
        self.load_or_train()
    
    def load_or_train(self):
        if os.path.exists("sentiment_model.pkl") and os.path.exists("vectorizer.pkl"):
            with open("sentiment_model.pkl", "rb") as f:
                self.model = pickle.load(f)
            with open("vectorizer.pkl", "rb") as f:
                self.vectorizer = pickle.load(f)
            self.trained = True
            logger.info("Модель тональності завантажена")
        else:
            self.train_from_bogon()
    
    def train_from_bogon(self):
        if not os.path.exists(BOGON_PATH):
            # Створюємо базовий bogon.json
            default_data = {
                "negative": [
                    "погано", "жахливо", "відстій", "ненавиджу", "дурня", "ідіот",
                    "плохо", "ужасно", "отстой", "ненавижу", "дурак", "идиот"
                ],
                "positive": [
                    "чудово", "супер", "класно", "люблю", "прекрасно", "найкраще",
                    "отлично", "супер", "классно", "люблю", "прекрасно", "лучшее"
                ],
                "neutral": [
                    "нормально", "окей", "добре", "розумію", "так", "ні",
                    "нормально", "окей", "хорошо", "понимаю", "да", "нет"
                ]
            }
            with open(BOGON_PATH, "w", encoding="utf-8") as f:
                json.dump(default_data, f, ensure_ascii=False, indent=2)
        
        with open(BOGON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        texts = []
        labels = []
        
        for sentiment, examples in data.items():
            texts.extend(examples)
            labels.extend([sentiment] * len(examples))
        
        if len(texts) > 0:
            X = self.vectorizer.fit_transform(texts)
            self.model.fit(X, labels)
            self.trained = True
            
            with open("sentiment_model.pkl", "wb") as f:
                pickle.dump(self.model, f)
            with open("vectorizer.pkl", "wb") as f:
                pickle.dump(self.vectorizer, f)
            
            logger.info(f"Модель навчена на {len(texts)} прикладах")
    
    def predict(self, text: str) -> tuple:
        if not self.trained:
            return ("neutral", 0.5)
        
        text_clean = re.sub(r'[^\w\s]', '', text.lower())
        X = self.vectorizer.transform([text_clean])
        prediction = self.model.predict(X)[0]
        proba = self.model.predict_proba(X)[0]
        confidence = max(proba)
        
        return (prediction, confidence)
    
    def retrain(self, new_texts: List[str], new_labels: List[str]):
        # Додавання нових даних до bogon.json
        with open(BOGON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        for text, label in zip(new_texts, new_labels):
            if label in data:
                data[label].append(text)
        
        with open(BOGON_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        self.train_from_bogon()

sentiment_analyzer = SentimentAnalyzer()

# ==================== SCRAPERS ====================

# TELEGRAM SCRAPER
class TelegramScraper:
    """Telegram scraper з підтримкою публічних та приватних каналів"""
    
    def __init__(self):
        self.client = None
        self.session_file = os.path.join(SESSION_DIR, "telegram.session")
    
    async def init_client(self, api_id: str, api_hash: str):
        """Ініціалізація Telegram клієнта"""
        try:
            self.client = TelegramClient(self.session_file, int(api_id), api_hash)
            await self.client.start()
            logger.info("✓ Telegram клієнт успішно ініціалізовано")
        except Exception as e:
            logger.error(f"✗ Помилка ініціалізації Telegram: {e}")
            raise
    
    async def scrape_post(self, url: str) -> Dict:
        """Збір даних з Telegram поста"""
        if not self.client:
            logger.error("✗ Telegram клієнт не ініціалізовано")
            return {"views": 0, "likes": 0, "comments": [], "reposts": 0, "error": "Client not initialized"}
        
        try:
            logger.info(f"✈️ Telegram парсинг: {url}")
            url = url.strip()
            
            # Парсинг URL
            channel_id = None
            message_id = None
            
            if "/c/" in url:
                # Приватний канал
                parts = url.split("/c/")[1].split("/")
                channel_id = int("-100" + parts[0])
                message_id = int(parts[1])
                logger.info(f"✓ Приватний канал: ID={channel_id}, message={message_id}")
                
            else:
                # Публічний канал
                parts = url.replace("https://t.me/", "").replace("http://t.me/", "").split("/")
                channel = parts[0]
                message_id = int(parts[1]) if len(parts) > 1 else None
                
                if not message_id:
                    logger.error(f"✗ Не вдалося визначити ID повідомлення з URL: {url}")
                    return {"views": 0, "likes": 0, "comments": [], "reposts": 0, "error": "Invalid URL"}
                
                channel_id = channel
                logger.info(f"✓ Публічний канал: @{channel}, message={message_id}")
            
            # Отримання entity
            try:
                entity = await self.client.get_entity(channel_id)
                logger.info(f"✓ Entity отримано: {entity.title if hasattr(entity, 'title') else entity.username}")
            except Exception as e:
                logger.error(f"✗ Не вдалося отримати entity: {e}")
                return {"views": 0, "likes": 0, "comments": [], "reposts": 0, "error": f"Entity error: {str(e)}"}
            
            # Отримання повідомлення
            try:
                message = await self.client.get_messages(entity, ids=message_id)
            except Exception as e:
                logger.error(f"✗ Не вдалося отримати повідомлення: {e}")
                return {"views": 0, "likes": 0, "comments": [], "reposts": 0, "error": f"Message error: {str(e)}"}
            
            if not message:
                logger.error(f"✗ Повідомлення {message_id} не знайдено")
                return {"views": 0, "likes": 0, "comments": [], "reposts": 0, "error": "Message not found"}
            
            logger.info(f"✓ Повідомлення отримано: ID={message.id}, перегляди={message.views}")
            
            # Метрики
            comments = []
            likes = 0
            
            # Реакції
            if message.reactions and message.reactions.results:
                likes = sum(r.count for r in message.reactions.results)
                logger.info(f"✓ Реакції: {likes}")
            
            # ЗБІР КОМЕНТАРІВ
            if hasattr(message, 'replies') and message.replies and message.replies.replies > 0:
                logger.info(f"💬 Збір коментарів (до {min(message.replies.replies, 100)})...")
                
                try:
                    # Метод 1: Стандартний iter_messages
                    async for msg in self.client.iter_messages(
                        entity, 
                        reply_to=message_id, 
                        limit=100
                    ):
                        if msg.text:
                            author = "Anonymous"
                            
                            # Спроба отримати автора
                            try:
                                if msg.sender:
                                    if hasattr(msg.sender, 'username') and msg.sender.username:
                                        author = msg.sender.username
                                    elif hasattr(msg.sender, 'first_name'):
                                        author = msg.sender.first_name
                            except:
                                pass
                            
                            comments.append({
                                "text": msg.text,
                                "author": author
                            })
                    
                    logger.info(f"✅ Зібрано {len(comments)} коментарів")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Не вдалося отримати коментарі: {e}")
                    
                    # Метод 2: Через discussion group (якщо є)
                    if hasattr(message.replies, 'channel_id') and message.replies.channel_id:
                        try:
                            logger.info("🔄 Спроба через discussion group...")
                            
                            discussion_entity = await self.client.get_entity(message.replies.channel_id)
                            
                            async for msg in self.client.iter_messages(
                                discussion_entity,
                                limit=100
                            ):
                                if msg.text and len(msg.text) > 3:
                                    author = "Anonymous"
                                    
                                    try:
                                        if msg.sender:
                                            if hasattr(msg.sender, 'username') and msg.sender.username:
                                                author = msg.sender.username
                                            elif hasattr(msg.sender, 'first_name'):
                                                author = msg.sender.first_name
                                    except:
                                        pass
                                    
                                    comments.append({
                                        "text": msg.text,
                                        "author": author
                                    })
                            
                            logger.info(f"✅ Зібрано {len(comments)} коментарів через discussion group")
                            
                        except Exception as e2:
                            logger.warning(f"⚠️ Discussion group теж не спрацював: {e2}")
            else:
                logger.info("ℹ️ Коментарі відсутні або відключені")
            
            return {
                "views": message.views or 0,
                "likes": likes,
                "comments": comments,
                "reposts": message.forwards or 0
            }
            
        except ValueError as e:
            logger.error(f"✗ Помилка парсингу URL {url}: {e}")
            return {"views": 0, "likes": 0, "comments": [], "reposts": 0, "error": str(e)}
        except Exception as e:
            logger.error(f"✗ Telegram scraper error для {url}: {e}", exc_info=True)
            return {"views": 0, "likes": 0, "comments": [], "reposts": 0, "error": str(e)}


# INSTAGRAM SCRAPER
class InstagramScraper:
    """Instagram scraper з покращеним збором коментарів"""
    
    def __init__(self):
        self.client = None
        self.session_file = os.path.join(SESSION_DIR, "instagram.json")

    def init_client(self, username: str, password: str):
        """Ініціалізація клієнта Instagram"""
        try:
            from instagrapi import Client as InstaClient
            self.client = InstaClient()

            # Завантаження існуючої сесії
            if os.path.exists(self.session_file):
                try:
                    self.client.load_settings(self.session_file)
                    self.client.login(username, password)
                    logger.info("✓ Instagram сесія відновлена")
                    return
                except Exception as e:
                    logger.warning(f"⚠️ Не вдалося відновити сесію: {e}")

            # Новий логін
            self.client.login(username, password)
            self.client.dump_settings(self.session_file)
            logger.info("✓ Instagram клієнт успішно авторизовано")

        except Exception as e:
            logger.error(f"✗ Instagram login error: {e}", exc_info=True)
            self.client = None

    def scrape_post(self, url: str) -> Dict:
        """Збір даних з Instagram поста"""
        if not self.client:
            logger.error("✗ Instagram клієнт не авторизовано")
            return {"views": 0, "likes": 0, "comments": [], "reposts": 0, "error": "Not authenticated"}

        try:
            logger.info(f"📷 Instagram парсинг: {url}")
            media_pk = self.client.media_pk_from_url(url)
            logger.info(f"✓ Media PK: {media_pk}")

            # Отримання метрик
            like_count = 0
            view_count = 0
            comment_count = 0
            
            try:
                # Спроба 1: Стандартний метод
                media = self.client.media_info(media_pk)
                like_count = getattr(media, "like_count", 0)
                view_count = getattr(media, "view_count", 0)
                comment_count = getattr(media, "comment_count", 0)
                
            except Exception as e:
                logger.warning(f"⚠️ media_info() впав: {e}")
                
                try:
                    # Спроба 2: Ручний запит
                    raw = self.client.private_request(f"media/{media_pk}/info/")
                    if "items" in raw and raw["items"]:
                        item = raw["items"][0]
                        like_count = item.get("like_count", 0)
                        view_count = item.get("play_count", 0) or item.get("view_count", 0)
                        comment_count = item.get("comment_count", 0)
                        logger.info("✓ Метрики отримано через raw запит")
                except Exception as e2:
                    logger.error(f"✗ Raw запит теж впав: {e2}")
                    return {"views": 0, "likes": 0, "comments": [], "reposts": 0, "error": str(e2)}

            logger.info(f"✓ Метрики: likes={like_count}, views={view_count}, comments={comment_count}")

            # ЗБІР КОМЕНТАРІВ - ПОКРАЩЕНИЙ
            comments = []
            
            if comment_count > 0:
                logger.info(f"💬 Збір коментарів (до {min(comment_count, 100)})...")
                
                try:
                    # Метод 1: Стандартний media_comments
                    try:
                        comments_data = self.client.media_comments(media_pk, amount=100)
                        
                        for comment in comments_data:
                            try:
                                text = comment.text if hasattr(comment, 'text') else str(comment)
                                author = comment.user.username if hasattr(comment, 'user') else "Instagram User"
                                
                                if 3 <= len(text) <= 500:
                                    comments.append({
                                        "text": text,
                                        "author": author
                                    })
                            except:
                                continue
                        
                        logger.info(f"✓ Зібрано {len(comments)} коментарів через media_comments()")
                        
                    except Exception as e:
                        logger.warning(f"⚠️ media_comments() не спрацював: {e}")
                    
                    # Метод 2: Ручний запит (якщо перший метод не дав результатів)
                    if len(comments) < 5:
                        logger.info("🔄 Спроба через raw API...")
                        
                        try:
                            raw_comments = self.client.private_request(
                                f"media/{media_pk}/comments/?can_support_threading=true&permalink_enabled=false"
                            )
                            
                            if "comments" in raw_comments:
                                for c in raw_comments["comments"]:
                                    text = c.get("text", "")
                                    author = c.get("user", {}).get("username", "Instagram User")
                                    
                                    if 3 <= len(text) <= 500:
                                        if not any(comment["text"] == text for comment in comments):
                                            comments.append({
                                                "text": text,
                                                "author": author
                                            })
                                
                                logger.info(f"✓ Зібрано {len(comments)} коментарів через raw API")
                                
                        except Exception as e:
                            logger.warning(f"⚠️ Raw API не спрацював: {e}")
                
                except Exception as e:
                    logger.error(f"✗ Помилка збору коментарів: {e}", exc_info=True)

            return {
                "views": view_count or 0,
                "likes": like_count or 0,
                "comments": comments[:100],
                "reposts": 0,
            }

        except Exception as e:
            logger.error(f"✗ Instagram scraper error для {url}: {e}", exc_info=True)
            return {"views": 0, "likes": 0, "comments": [], "reposts": 0, "error": str(e)}

# YOUTUBE SCRAPER
class YouTubeScraper:
    """YouTube scraper через офіційне API"""
    
    def __init__(self, api_key: str):
        try:
            from googleapiclient.discovery import build
            self.youtube = build('youtube', 'v3', developerKey=api_key)
            logger.info("✓ YouTube API клієнт ініціалізовано")
        except Exception as e:
            logger.error(f"✗ Помилка ініціалізації YouTube API: {e}")
            self.youtube = None
    
    def scrape_video(self, url: str) -> Dict:
        """Збір даних з YouTube відео"""
        if not self.youtube:
            logger.error("✗ YouTube API не ініціалізовано")
            return {"views": 0, "likes": 0, "comments": [], "reposts": 0, "error": "API not initialized"}
        
        try:
            logger.info(f"🎥 YouTube парсинг: {url}")
            
            # Парсинг video_id
            if "v=" in url:
                video_id = url.split("v=")[1].split("&")[0]
            elif "youtu.be/" in url:
                video_id = url.split("youtu.be/")[1].split("?")[0]
            else:
                logger.error(f"✗ Невідомий формат URL: {url}")
                return {"views": 0, "likes": 0, "comments": [], "reposts": 0, "error": "Invalid URL"}
            
            logger.info(f"✓ Video ID: {video_id}")
            
            # Отримання статистики відео
            video_response = self.youtube.videos().list(
                part="statistics,snippet",
                id=video_id
            ).execute()
            
            if not video_response.get('items'):
                logger.error(f"✗ Відео не знайдено: {video_id}")
                return {"views": 0, "likes": 0, "comments": [], "reposts": 0, "error": "Video not found"}
            
            stats = video_response['items'][0]['statistics']
            snippet = video_response['items'][0]['snippet']
            
            logger.info(f"✓ Перегляди: {stats.get('viewCount')}, Лайки: {stats.get('likeCount')}")
            
            # Збір коментарів
            comments = []
            if snippet.get('liveBroadcastContent') != 'live' and stats.get('commentCount', '0') != '0':
                try:
                    comments_response = self.youtube.commentThreads().list(
                        part="snippet",
                        videoId=video_id,
                        maxResults=100,
                        order="relevance"
                    ).execute()
                    
                    for item in comments_response.get('items', []):
                        comment = item['snippet']['topLevelComment']['snippet']
                        comments.append({
                            "text": comment['textDisplay'],
                            "author": comment['authorDisplayName']
                        })
                    
                    logger.info(f"✅ Зібрано {len(comments)} коментарів")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Не вдалося отримати коментарі: {e}")
            else:
                logger.info("ℹ️ Коментарі відключені або відео в прямому ефірі")
            
            return {
                "views": int(stats.get('viewCount', 0)),
                "likes": int(stats.get('likeCount', 0)),
                "comments": comments,
                "reposts": 0
            }
            
        except Exception as e:
            logger.error(f"✗ YouTube scraper error для {url}: {e}", exc_info=True)
            return {"views": 0, "likes": 0, "comments": [], "reposts": 0, "error": str(e)}


# ==================== ПОКРАЩЕНА ФУНКЦІЯ SCRAPING ====================
async def scrape_all_sources():
    """Фонове завдання для збору даних з усіх джерел з retry-логікою"""
    logger.info("=== Початок циклу збору даних ===")
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    try:
        sources = c.execute("SELECT id, url, platform FROM sources WHERE active=1").fetchall()
        logger.info(f"Знайдено активних джерел: {len(sources)}")
        
        if not sources:
            logger.warning("Немає активних джерел для парсингу")
            return
        
        # Отримання API ключів
        api_keys = {}
        keys_data = c.execute("SELECT platform, key_name, key_value FROM api_keys").fetchall()
        
        for platform, key_name, encrypted_value in keys_data:
            if platform not in api_keys:
                api_keys[platform] = {}
            try:
                decrypted_value = fernet.decrypt(encrypted_value.encode()).decode()
                api_keys[platform][key_name] = decrypted_value
            except Exception as e:
                logger.error(f"Помилка розшифрування ключа {platform}/{key_name}: {e}")
        
        logger.info(f"Завантажено ключів для платформ: {list(api_keys.keys())}")
        
        # Ініціалізація scrapers
        telegram_scraper = TelegramScraper()
        instagram_scraper = InstagramScraper()
        youtube_scraper = YouTubeScraper()

        # Telegram
        if "telegram" in api_keys and "api_id" in api_keys["telegram"] and "api_hash" in api_keys["telegram"]:
            telegram_scraper = TelegramScraper()
            try:
                await telegram_scraper.init_client(
                    api_keys["telegram"]["api_id"],
                    api_keys["telegram"]["api_hash"]
                )
            except Exception as e:
                logger.error(f"Не вдалося ініціалізувати Telegram: {e}")
        else:
            logger.warning("Telegram API ключі не налаштовані")
        
        # Instagram
        if "instagram" in api_keys and "username" in api_keys["instagram"] and "password" in api_keys["instagram"]:
            instagram_scraper = InstagramScraper()
            try:
                instagram_scraper.init_client(
                    api_keys["instagram"]["username"],
                    api_keys["instagram"]["password"]
                )
            except Exception as e:
                logger.error(f"Не вдалося ініціалізувати Instagram: {e}")
        else:
            logger.warning("Instagram API ключі не налаштовані")
        
        # YouTube
        if "youtube" in api_keys and "api_key" in api_keys["youtube"]:
            youtube_scraper = YouTubeScraper(api_keys["youtube"]["api_key"])
        else:
            logger.warning("YouTube API ключ не налаштований")
        
        # Обробка кожного джерела з retry
        max_retries = 2
        
        for source_id, url, platform in sources:
            logger.info(f"\n--- Обробка джерела {source_id} ({platform}): {url} ---")
            
            data = None
            last_error = None
            
            # Retry логіка
            for attempt in range(max_retries):
                try:
                    if attempt > 0:
                        logger.info(f"Спроба {attempt + 1}/{max_retries}")
                        await asyncio.sleep(5 * attempt)
                    
                    if platform == "telegram":
                        if telegram_scraper:
                            data = await telegram_scraper.scrape_post(url)
                        else:
                            logger.error("Telegram scraper не ініціалізовано")
                            break
                    
                    elif platform == "instagram":
                        if instagram_scraper:
                            data = instagram_scraper.scrape_post(url)
                        else:
                            logger.error("Instagram scraper не ініціалізовано")
                            break
                    
                    elif platform == "youtube":
                        if youtube_scraper:
                            data = youtube_scraper.scrape_video(url)
                        else:
                            logger.error("YouTube scraper не ініціалізовано")
                            break
                    
                    else:
                        logger.warning(f"Невідома платформа: {platform}")
                        break
                    
                    # Якщо отримали дані - виходимо з retry циклу
                    if data and "error" not in data:
                        break
                    elif data and "error" in data:
                        last_error = data["error"]
                        if any(x in last_error.lower() for x in ['auth', 'login', 'credential', 'password', 'timeout']):
                            logger.error(f"Помилка {last_error} - пропускаємо retry")
                            break
                
                except Exception as e:
                    last_error = str(e)
                    logger.error(f"Спроба {attempt + 1} не вдалась: {e}")
                    if attempt == max_retries - 1:
                        logger.error(f"Всі спроби вичерпано для джерела {source_id}")
            
            # Збереження результатів
            if data and "error" not in data and (data.get("views", 0) > 0 or data.get("likes", 0) > 0 or len(data.get("comments", [])) > 0):
                try:
                    negative_count = 0
                    positive_count = 0
                    neutral_count = 0
                    
                    # Аналіз коментарів
                    comments_saved = 0
                    for comment in data.get("comments", []):
                        try:
                            sentiment, confidence = sentiment_analyzer.predict(comment["text"])
                            
                            if sentiment == "negative":
                                negative_count += 1
                            elif sentiment == "positive":
                                positive_count += 1
                            else:
                                neutral_count += 1
                            
                            c.execute(
                                "INSERT INTO comments (source_id, text, sentiment, confidence, author) VALUES (?, ?, ?, ?, ?)",
                                (source_id, comment["text"], sentiment, confidence, comment.get("author", ""))
                            )
                            comments_saved += 1
                        except Exception as e:
                            logger.error(f"Помилка збереження коментар: {e}")
                    
                    # Збереження загальних метрик
                    c.execute(
                        "INSERT INTO metrics (source_id, views, likes, comments_count, reposts, negative_count) VALUES (?, ?, ?, ?, ?, ?)",
                        (source_id, data.get("views", 0), data.get("likes", 0), len(data.get("comments", [])), data.get("reposts", 0), negative_count)
                    )
                    
                    conn.commit()
                    
                    logger.info(f"✓ Успішно збережено: перегляди={data.get('views', 0)}, лайки={data.get('likes', 0)}, коментарів={comments_saved}")
                    logger.info(f"  Тональність: позитив={positive_count}, негатив={negative_count}, нейтрал={neutral_count}")
                
                except Exception as e:
                    logger.error(f"Помилка збереження даних: {e}", exc_info=True)
            
            elif data and "error" in data:
                logger.error(f"Помилка парсингу: {data['error']}")
            else:
                logger.warning("Не отримано даних від scraper або всі метрики = 0")
            
            # Пауза між запитами
            delay = 5 if platform == "facebook" else 2
            await asyncio.sleep(delay)
        
        logger.info("=== Цикл збору даних завершено ===")
    
    except Exception as e:
        logger.error(f"Критична помилка в scrape_all_sources: {e}", exc_info=True)
    
    finally:
        conn.close()


# ==================== SCHEDULER ====================
@app.on_event("startup")
async def startup_event():
    """Ініціалізація планувальника при старті сервера"""
    try:
        existing_jobs = scheduler.get_jobs()
        job_ids = [job.id for job in existing_jobs]
        
        if 'scrape_job' not in job_ids:
            scheduler.add_job(
                scrape_all_sources, 
                'interval', 
                minutes=30, 
                id='scrape_job',
                replace_existing=True
            )
            logger.info("Scheduler job 'scrape_job' додано")
        else:
            logger.info("Scheduler job 'scrape_job' вже існує")
        
        scheduler.start()
        logger.info("Scheduler запущено успішно")
        
    except Exception as e:
        logger.error(f"Помилка запуску scheduler: {e}", exc_info=True)


@app.on_event("shutdown")
async def shutdown_event():
    """Зупинка планувальника при вимкненні сервера"""
    try:
        scheduler.shutdown(wait=False)
        logger.info("Scheduler зупинено")
    except Exception as e:
        logger.error(f"Помилка зупинки scheduler: {e}")


# ==================== AUTHORIZATION ====================
def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    password_hash = hashlib.sha256(credentials.password.encode()).hexdigest()
    user = c.execute("SELECT username, role FROM users WHERE username=? AND password=?",
                    (credentials.username, password_hash)).fetchone()
    conn.close()
    
    if not user:
        raise HTTPException(status_code=401, detail="Невірні облікові дані")
    
    return {"username": user[0], "role": user[1]}

def admin_required(user: dict = Depends(verify_credentials)):
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Потрібні права адміністратора")
    return user

# ==================== API ENDPOINTS ====================
@app.get("/", response_class=HTMLResponse)
async def root():
    html = """<!DOCTYPE html>
<html lang="uk">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Моніторинг соціальних мереж</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; padding: 20px; }
        .container { max-width: 1400px; margin: 0 auto; }
        .header { background: white; border-radius: 15px; padding: 30px; margin-bottom: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); }
        .header h1 { color: #667eea; margin-bottom: 10px; }
        .tabs { display: flex; gap: 10px; margin-bottom: 20px; flex-wrap: wrap; }
        .tab { background: white; border: none; padding: 15px 30px; border-radius: 10px; cursor: pointer; font-size: 16px; transition: all 0.3s; }
        .tab.active { background: #667eea; color: white; }
        .tab:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(0,0,0,0.2); }
        .content { background: white; border-radius: 15px; padding: 30px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); display: none; }
        .content.active { display: block; }
        .form-group { margin-bottom: 20px; }
        .form-group label { display: block; margin-bottom: 8px; font-weight: 600; color: #333; }
        .form-group input, .form-group select, .form-group textarea { width: 100%; padding: 12px; border: 2px solid #e0e0e0; border-radius: 8px; font-size: 14px; transition: border 0.3s; }
        .form-group input:focus, .form-group select:focus, .form-group textarea:focus { outline: none; border-color: #667eea; }
        .btn { background: #667eea; color: white; border: none; padding: 12px 30px; border-radius: 8px; cursor: pointer; font-size: 16px; transition: all 0.3s; }
        .btn:hover { background: #5568d3; transform: translateY(-2px); box-shadow: 0 5px 15px rgba(102,126,234,0.4); }
        .btn-danger { background: #e74c3c; }
        .btn-danger:hover { background: #c0392b; }
        .table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        .table th { background: #667eea; color: white; padding: 15px; text-align: left; }
        .table td { padding: 12px; border-bottom: 1px solid #e0e0e0; }
        .table tr:hover { background: #f8f9fa; }
        .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .stat-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 25px; border-radius: 15px; box-shadow: 0 5px 15px rgba(0,0,0,0.2); }
        .stat-card h3 { font-size: 14px; opacity: 0.9; margin-bottom: 10px; }
        .stat-card .number { font-size: 36px; font-weight: bold; }
        .sentiment-badge { padding: 5px 12px; border-radius: 20px; font-size: 12px; font-weight: 600; }
        .sentiment-negative { background: #e74c3c; color: white; }
        .sentiment-positive { background: #2ecc71; color: white; }
        .sentiment-neutral { background: #95a5a6; color: white; }
        .login-form { max-width: 400px; margin: 100px auto; background: white; padding: 40px; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.2); }
        .login-form h2 { color: #667eea; margin-bottom: 30px; text-align: center; }
        .alert { padding: 15px; border-radius: 8px; margin-bottom: 20px; }
        .alert-success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .alert-error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
    </style>
</head>
<body>
    <div class="container" id="app">
        <div class="header">
            <h1>📊 Моніторинг соціальних мереж</h1>
            <p>Автоматичний збір та аналіз контенту з Instagram, Telegram, YouTube</p>
            <div style="margin-top: 15px;">
                <span id="userInfo"></span>
                <button class="btn btn-danger" onclick="logout()" style="float: right;">Вийти</button>
            </div>
        </div>

        <div class="tabs">
            <button class="tab active" onclick="showTab('dashboard')">📈 Дашборд</button>
            <button class="tab" onclick="showTab('sources')">🔗 Джерела</button>
            <button class="tab" onclick="showTab('comments')">💬 Коментарі</button>
            <button class="tab" onclick="showTab('training')">🧠 Навчання ШІ</button>
            <button class="tab" onclick="showTab('api-keys')" id="apiKeysTab" style="display:none;">🔑 API Ключі</button>
        </div>

        <div id="dashboard" class="content active">
            <h2>Загальна статистика</h2>
            <div class="stats">
                <div class="stat-card">
                    <h3>Всього джерел</h3>
                    <div class="number" id="totalSources">0</div>
                </div>
                <div class="stat-card">
                    <h3>Всього коментарів</h3>
                    <div class="number" id="totalComments">0</div>
                </div>
                <div class="stat-card">
                    <h3>Негативних</h3>
                    <div class="number" id="negativeComments">0</div>
                </div>
                <div class="stat-card">
                    <h3>Позитивних</h3>
                    <div class="number" id="positiveComments">0</div>
                </div>
            </div>
            <button class="btn" onclick="refreshStats()">🔄 Оновити статистику</button>
            <button class="btn" onclick="runManualScrape()" style="margin-left: 10px; background: #e74c3c;">▶️ Запустити парсинг</button>
        </div>

        <div id="sources" class="content">
            <h2>Управління джерелами</h2>
            <div class="form-group">
                <label>URL посилання</label>
                <input type="text" id="sourceUrl" placeholder="https://...">
            </div>
            <div class="form-group">
                <label>Платформа</label>
                <select id="sourcePlatform">
                    <option value="telegram">Telegram</option>
                    <option value="instagram">Instagram</option>
                    <option value="youtube">YouTube</option>
                </select>
            </div>
            <div class="form-group">
                <label>Назва (необов'язково)</label>
                <input type="text" id="sourceTitle">
            </div>
            <button class="btn" onclick="addSource()">➕ Додати джерело</button>
            
            <table class="table">
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>URL</th>
                        <th>Платформа</th>
                        <th>Статус</th>
                        <th>Дії</th>
                    </tr>
                </thead>
                <tbody id="sourcesTable"></tbody>
            </table>
        </div>

        <div id="comments" class="content">
            <h2>Коментарі з моніторингу</h2>
            <div class="form-group">
                <label>Фільтр тональності</label>
                <select id="sentimentFilter" onchange="loadComments()">
                    <option value="">Всі</option>
                    <option value="negative">Негативні</option>
                    <option value="positive">Позитивні</option>
                    <option value="neutral">Нейтральні</option>
                </select>
            </div>
            <table class="table">
                <thead>
                    <tr>
                        <th>Автор</th>
                        <th>Текст</th>
                        <th>Тональність</th>
                        <th>Впевненість</th>
                        <th>Дата</th>
                    </tr>
                </thead>
                <tbody id="commentsTable"></tbody>
            </table>
        </div>

        <div id="training" class="content">
            <h2>Навчання моделі тональності</h2>
            <p style="margin-bottom: 20px;">Додайте приклади для покращення точності визначення тональності</p>
            
            <div class="form-group">
                <label>Текст коментаря</label>
                <textarea id="trainingText" rows="4" placeholder="Введіть текст коментаря..."></textarea>
            </div>
            <div class="form-group">
                <label>Тональність</label>
                <select id="trainingSentiment">
                    <option value="positive">Позитивна</option>
                    <option value="neutral">Нейтральна</option>
                    <option value="negative">Негативна</option>
                </select>
            </div>
            <button class="btn" onclick="addTrainingExample()">➕ Додати приклад</button>
            <button class="btn" onclick="retrainModel()" style="margin-left: 10px;">🔄 Перенавчити модель</button>
            
            <div id="trainingExamples" style="margin-top: 30px;">
                <h3>Додані приклади</h3>
                <div id="examplesList"></div>
            </div>
        </div>

        <div id="api-keys" class="content">
            <h2>Управління API ключами</h2>
            <p style="color: #e74c3c; margin-bottom: 20px;">⚠️ Тільки для адміністраторів</p>
            
            <h3>Telegram</h3>
            <div class="form-group">
                <label>API ID</label>
                <input type="text" id="telegramApiId">
            </div>
            <div class="form-group">
                <label>API Hash</label>
                <input type="text" id="telegramApiHash">
            </div>
            <button class="btn" onclick="saveApiKey('telegram')">💾 Зберегти</button>
            
            <h3 style="margin-top: 30px;">Instagram</h3>
            <div class="form-group">
                <label>Username</label>
                <input type="text" id="instagramUsername">
            </div>
            <div class="form-group">
                <label>Password</label>
                <input type="password" id="instagramPassword">
            </div>
            <button class="btn" onclick="saveApiKey('instagram')">💾 Зберегти</button>
            
            <h3 style="margin-top: 30px;">YouTube</h3>
            <div class="form-group">
                <label>API Key</label>
                <input type="text" id="youtubeApiKey">
            </div>
            <button class="btn" onclick="saveApiKey('youtube')">💾 Зберегти</button>
            
            <h3 style="margin-top: 30px;">Facebook</h3>
            <div class="form-group">
                <label>Email (для авторизації)</label>
                <input type="email" id="facebookEmail">
            </div>
            <div class="form-group">
                <label>Password</label>
                <input type="password" id="facebookPassword">
            </div>
            <button class="btn" onclick="saveApiKey('facebook')">💾 Зберегти</button>
        </div>
    </div>

    <script>
        let currentUser = null;
        let trainingExamples = [];

        async function runManualScrape() {
            if (!confirm('Запустити парсинг всіх джерел зараз?')) return;
            
            try {
                document.querySelector('.container').style.opacity = '0.5';
                const response = await apiCall('/api/scrape/manual', { method: 'POST' });
                const result = await response.json();
                
                if (response.ok) {
                    alert('✓ ' + result.message);
                    loadDashboard();
                } else {
                    alert('✗ Помилка: ' + (result.detail || 'Невідома помилка'));
                }
            } catch (e) {
                alert('✗ Помилка запуску парсингу: ' + e.message);
            } finally {
                document.querySelector('.container').style.opacity = '1';
            }
        }

        // Перевірка авторизації
        async function checkAuth() {
            try {
                const response = await fetch('/api/user', {
                    headers: { 'Authorization': 'Basic ' + btoa(localStorage.getItem('username') + ':' + localStorage.getItem('password')) }
                });
                if (response.ok) {
                    currentUser = await response.json();
                    document.getElementById('userInfo').textContent = `Користувач: ${currentUser.username} (${currentUser.role})`;
                    if (currentUser.role === 'admin') {
                        document.getElementById('apiKeysTab').style.display = 'block';
                    }
                    loadDashboard();
                } else {
                    showLogin();
                }
            } catch (e) {
                showLogin();
            }
        }

        function showLogin() {
            document.querySelector('.container').innerHTML = `
                <div class="login-form">
                    <h2>Вхід до системи</h2>
                    <div class="form-group">
                        <label>Логін</label>
                        <input type="text" id="loginUsername" value="admin">
                    </div>
                    <div class="form-group">
                        <label>Пароль</label>
                        <input type="password" id="loginPassword" value="admin123">
                    </div>
                    <button class="btn" onclick="login()" style="width: 100%;">Увійти</button>
                    <div id="loginError"></div>
                </div>
            `;
        }

        async function login() {
            const username = document.getElementById('loginUsername').value;
            const password = document.getElementById('loginPassword').value;
            
            const response = await fetch('/api/user', {
                headers: { 'Authorization': 'Basic ' + btoa(username + ':' + password) }
            });
            
            if (response.ok) {
                localStorage.setItem('username', username);
                localStorage.setItem('password', password);
                location.reload();
            } else {
                document.getElementById('loginError').innerHTML = '<div class="alert alert-error">Невірні облікові дані</div>';
            }
        }

        function logout() {
            localStorage.removeItem('username');
            localStorage.removeItem('password');
            location.reload();
        }

        function showTab(tabName) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.content').forEach(c => c.classList.remove('active'));
            event.target.classList.add('active');
            document.getElementById(tabName).classList.add('active');
            
            if (tabName === 'dashboard') loadDashboard();
            if (tabName === 'sources') loadSources();
            if (tabName === 'comments') loadComments();
        }

        async function apiCall(endpoint, options = {}) {
            const headers = {
                'Authorization': 'Basic ' + btoa(localStorage.getItem('username') + ':' + localStorage.getItem('password')),
                'Content-Type': 'application/json',
                ...options.headers
            };
            return fetch(endpoint, { ...options, headers });
        }

        async function loadDashboard() {
            const stats = await (await apiCall('/api/stats')).json();
            document.getElementById('totalSources').textContent = stats.total_sources;
            document.getElementById('totalComments').textContent = stats.total_comments;
            document.getElementById('negativeComments').textContent = stats.negative_comments;
            document.getElementById('positiveComments').textContent = stats.positive_comments;
        }

        async function addSource() {
            const url = document.getElementById('sourceUrl').value;
            const platform = document.getElementById('sourcePlatform').value;
            const title = document.getElementById('sourceTitle').value;
            
            const response = await apiCall('/api/sources', {
                method: 'POST',
                body: JSON.stringify({ url, platform, title })
            });
            
            if (response.ok) {
                alert('Джерело додано успішно!');
                document.getElementById('sourceUrl').value = '';
                document.getElementById('sourceTitle').value = '';
                loadSources();
            } else {
                alert('Помилка додавання джерела');
            }
        }

        async function loadSources() {
            const sources = await (await apiCall('/api/sources')).json();
            const tbody = document.getElementById('sourcesTable');
            tbody.innerHTML = sources.map(s => `
                <tr>
                    <td>${s.id}</td>
                    <td>${s.url}</td>
                    <td>${s.platform}</td>
                    <td>${s.active ? '✅ Активне' : '❌ Неактивне'}</td>
                    <td>
                        <button class="btn" onclick="toggleSource(${s.id}, ${s.active})">
                            ${s.active ? 'Деактивувати' : 'Активувати'}
                        </button>
                        <button class="btn btn-danger" onclick="deleteSource(${s.id})">Видалити</button>
                    </td>
                </tr>
            `).join('');
        }

        async function toggleSource(id, active) {
            await apiCall(`/api/sources/${id}`, {
                method: 'PATCH',
                body: JSON.stringify({ active: !active })
            });
            loadSources();
        }

        async function deleteSource(id) {
            if (confirm('Видалити це джерело?')) {
                await apiCall(`/api/sources/${id}`, { method: 'DELETE' });
                loadSources();
            }
        }

        async function loadComments() {
            const filter = document.getElementById('sentimentFilter').value;
            const comments = await (await apiCall(`/api/comments?sentiment=${filter}`)).json();
            const tbody = document.getElementById('commentsTable');
            tbody.innerHTML = comments.map(c => `
                <tr>
                    <td>${c.author || 'Анонім'}</td>
                    <td>${c.text}</td>
                    <td><span class="sentiment-badge sentiment-${c.sentiment}">${c.sentiment}</span></td>
                    <td>${(c.confidence * 100).toFixed(1)}%</td>
                    <td>${new Date(c.collected_at).toLocaleString('uk-UA')}</td>
                </tr>
            `).join('');
        }

        async function addTrainingExample() {
            const text = document.getElementById('trainingText').value;
            const sentiment = document.getElementById('trainingSentiment').value;
            
            if (!text) {
                alert('Введіть текст коментаря');
                return;
            }
            
            trainingExamples.push({ text, sentiment });
            document.getElementById('trainingText').value = '';
            updateExamplesList();
        }

        function updateExamplesList() {
            const list = document.getElementById('examplesList');
            list.innerHTML = trainingExamples.map((ex, i) => `
                <div style="padding: 10px; background: #f8f9fa; margin: 10px 0; border-radius: 8px;">
                    <strong>${ex.sentiment}:</strong> ${ex.text}
                    <button class="btn btn-danger" onclick="removeExample(${i})" style="float: right; padding: 5px 10px;">Видалити</button>
                </div>
            `).join('');
        }

        function removeExample(index) {
            trainingExamples.splice(index, 1);
            updateExamplesList();
        }

        async function retrainModel() {
            if (trainingExamples.length === 0) {
                alert('Додайте хоча б один приклад');
                return;
            }
            
            const response = await apiCall('/api/train', {
                method: 'POST',
                body: JSON.stringify({ examples: trainingExamples })
            });
            
            if (response.ok) {
                alert('Модель успішно перенавчена!');
                trainingExamples = [];
                updateExamplesList();
            } else {
                alert('Помилка перенавчання моделі');
            }
        }

        async function saveApiKey(platform) {
            let data = {};
            
            if (platform === 'telegram') {
                data = {
                    api_id: document.getElementById('telegramApiId').value,
                    api_hash: document.getElementById('telegramApiHash').value
                };
            } else if (platform === 'instagram') {
                data = {
                    username: document.getElementById('instagramUsername').value,
                    password: document.getElementById('instagramPassword').value
                };
            } else if (platform === 'youtube') {
                data = {
                    api_key: document.getElementById('youtubeApiKey').value
                };
            } else if (platform === 'facebook') {
                data = {
                    email: document.getElementById('facebookEmail').value,
                    password: document.getElementById('facebookPassword').value
                };
            }
            
            const response = await apiCall(`/api/keys/${platform}`, {
                method: 'POST',
                body: JSON.stringify(data)
            });
            
            if (response.ok) {
                alert('API ключ збережено успішно!');
            } else {
                alert('Помилка збереження ключа');
            }
        }

        function refreshStats() {
            loadDashboard();
            alert('Статистика оновлена!');
        }

        // Ініціалізація при завантаженні
        checkAuth();
    </script>
</body>
</html>"""
    return html

@app.get("/api/user")
async def get_user(user: dict = Depends(verify_credentials)):
    return user

@app.get("/api/stats")
async def get_stats(user: dict = Depends(verify_credentials)):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    total_sources = c.execute("SELECT COUNT(*) FROM sources WHERE active=1").fetchone()[0]
    total_comments = c.execute("SELECT COUNT(*) FROM comments").fetchone()[0]
    negative_comments = c.execute("SELECT COUNT(*) FROM comments WHERE sentiment='negative'").fetchone()[0]
    positive_comments = c.execute("SELECT COUNT(*) FROM comments WHERE sentiment='positive'").fetchone()[0]
    
    conn.close()
    
    return {
        "total_sources": total_sources,
        "total_comments": total_comments,
        "negative_comments": negative_comments,
        "positive_comments": positive_comments
    }

class SourceCreate(BaseModel):
    url: str
    platform: str
    title: Optional[str] = None

@app.post("/api/sources")
async def create_source(source: SourceCreate, user: dict = Depends(verify_credentials)):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    try:
        c.execute("INSERT INTO sources (url, platform, title) VALUES (?, ?, ?)",
                 (source.url, source.platform, source.title))
        conn.commit()
        
        # Логування
        c.execute("INSERT INTO logs (user, action, details) VALUES (?, ?, ?)",
                 (user["username"], "add_source", f"Added {source.platform}: {source.url}"))
        conn.commit()
        
        conn.close()
        return {"success": True}
    except Exception as e:
        conn.close()
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/sources")
async def get_sources(user: dict = Depends(verify_credentials)):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    sources = c.execute("SELECT id, url, platform, title, active FROM sources ORDER BY id DESC").fetchall()
    conn.close()
    
    return [{"id": s[0], "url": s[1], "platform": s[2], "title": s[3], "active": s[4]} for s in sources]

@app.patch("/api/sources/{source_id}")
async def update_source(source_id: int, data: dict, user: dict = Depends(verify_credentials)):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    if "active" in data:
        c.execute("UPDATE sources SET active=? WHERE id=?", (data["active"], source_id))
    
    conn.commit()
    conn.close()
    return {"success": True}

@app.delete("/api/sources/{source_id}")
async def delete_source(source_id: int, user: dict = Depends(verify_credentials)):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM sources WHERE id=?", (source_id,))
    conn.commit()
    conn.close()
    return {"success": True}

@app.get("/api/comments")
async def get_comments(sentiment: str = "", user: dict = Depends(verify_credentials)):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    if sentiment:
        comments = c.execute(
            "SELECT text, sentiment, confidence, author, collected_at FROM comments WHERE sentiment=? ORDER BY id DESC LIMIT 100",
            (sentiment,)
        ).fetchall()
    else:
        comments = c.execute(
            "SELECT text, sentiment, confidence, author, collected_at FROM comments ORDER BY id DESC LIMIT 100"
        ).fetchall()
    
    conn.close()
    
    return [{"text": c[0], "sentiment": c[1], "confidence": c[2], "author": c[3], "collected_at": c[4]} for c in comments]

class TrainingData(BaseModel):
    examples: List[Dict[str, str]]

@app.post("/api/train")
async def train_model(data: TrainingData, user: dict = Depends(verify_credentials)):
    try:
        texts = [ex["text"] for ex in data.examples]
        labels = [ex["sentiment"] for ex in data.examples]
        sentiment_analyzer.retrain(texts, labels)
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/keys/{platform}")
async def save_api_key(platform: str, data: dict, user: dict = Depends(admin_required)):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    try:
        # Видалення старих ключів
        c.execute("DELETE FROM api_keys WHERE platform=?", (platform,))
        
        # Збереження нових ключів (зашифрованих)
        for key_name, key_value in data.items():
            encrypted_value = fernet.encrypt(str(key_value).encode()).decode()
            c.execute("INSERT INTO api_keys (platform, key_name, key_value) VALUES (?, ?, ?)",
                     (platform, key_name, encrypted_value))
        
        conn.commit()
        conn.close()
        return {"success": True}
    except Exception as e:
        conn.close()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/scrape/manual")
async def manual_scrape(user: dict = Depends(admin_required)):
    """Ручний запуск парсингу всіх джерел"""
    try:
        logger.info(f"Ручний запуск парсингу від користувача {user['username']}")
        await scrape_all_sources()
        return {"success": True, "message": "Парсинг завершено успішно"}
    except Exception as e:
        logger.error(f"Помилка ручного парсингу: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)