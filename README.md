# 📊 Social Media Monitoring System

README available in **English**, **Українська**, **Русский**.

---

# 🇺🇸 English

## 📌 Overview

This project is a **social media monitoring and analytics system** built on **FastAPI**, featuring:

* Automatic scraping of **Telegram**, **Instagram**, and **YouTube** posts
* Scheduled data collection (via APScheduler)
* Collection of views, likes, comments, reposts
* AI-powered **sentiment analysis** for comments
* Dashboard for statistics, sources management, comment review
* Retrainable ML model using user-provided examples
* Admin panel for managing API keys
* Authentication using HTTP Basic

---

## 🚀 Features

### ✅ Scrapers

* **Telegram Scraper** (public and private channels)
* **Instagram Scraper** with improved comments collection
* **YouTube Scraper** using official API

### 🤖 Sentiment Analyzer

* Trains automatically using `bogon.json`
* Supports retraining with custom examples

### 📊 Web Interface

* Dashboard with live statistics
* Source management (add / remove / activate / deactivate)
* Comments browser with sentiment filters
* Admin-only API keys management

### 🔐 Security

* Basic auth login
* Roles: *user* and *admin*
* API keys encrypted with **Fernet**

---

## 🛠️ Installation

```
pip install -r requirements.txt
python main.py
```

Open in browser: [http://localhost:8000](http://localhost:8000)

Default credentials:

```
Username: admin
Password: admin123
```

---

## 🧩 File Structure

* `main.py` – main application
* `db.db` – SQLite database
* `.session/` – saved sessions for scrapers
* `bogon.json` – sentiment training data
* `sentiment_model.pkl` – trained model

---

# 🇺🇦 Українська

## 📌 Опис

Це система **моніторингу соціальних мереж**, що автоматично збирає дані з:

* Telegram
* Instagram
* YouTube

Функціонал включає:

* Збір переглядів, лайків, коментарів, репостів
* Аналіз тональності коментарів (ШІ)
* Розклад парсингу
* Веб-інтерфейс з дашбордом
* Управління джерелами та API ключами
* Повторне навчання моделі

---

## 🚀 Можливості

### ✅ Скрейпери

* Telegram (публічні та приватні канали)
* Instagram з покращеним збором коментарів
* YouTube API

### 🤖 Аналіз тональності

* Автоматичне навчання
* Підтримка перенавчання з прикладами користувача

### 📊 Інтерфейс

* Дашборд статистики
* Таблиця джерел
* Перегляд коментарів з фільтрами
* Розділ навчання ШІ

### 🔐 Безпека

* Авторизація HTTP Basic
* Ролі: *користувач*, *адмін*
* API ключі зберігаються в зашифрованому вигляді

---

## 🛠️ Встановлення

```
pip install -r requirements.txt
python main.py
```

Відкрити: [http://localhost:8000](http://localhost:8000)

Стандартний логін:

```
admin / admin123
```

---

# 🇷🇺 Русский

## 📌 Описание

Это система **мониторинга социальных сетей**, которая автоматически собирает данные из:

* Telegram
* Instagram
* YouTube

Функционал включает:

* Сбор просмотров, лайков, комментариев, репостов
* AI-анализ тональности комментариев
* Планировщик автоматического сбора
* Веб-интерфейс с дашбордом
* Управление источниками и API‑ключами
* Переобучение ML‑модели

---

## 🚀 Возможности

### ✅ Скрейперы

* Telegram (публичные и приватные каналы)
* Instagram с улучшенным сбором комментариев
* YouTube через официальное API

### 🤖 Анализ тональности

* Автоматическое обучение на bogon.json
* Возможность добавлять свои обучающие примеры

### 📊 Интерфейс

* Дашборд статистики
* Управление источниками
* Просмотр комментариев
* Переобучение модели

### 🔐 Безопасность

* Авторизация HTTP Basic
* Роли: *user*, *admin*
* API‑ключи хранятся в зашифрованном виде

---

## 🛠️ Установка

```
pip install -r requirements.txt
python main.py
```

Открыть в браузере: [http://localhost:8000](http://localhost:8000)

Стандартный вход:

```
admin / admin123
```


readme.md - by ChatGPT
