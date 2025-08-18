# 📰 Daily AI News Feed

Welcome to **Daily AI News**, your personal, AI-curated digest of the latest **finance, startup, tech, and global news**.  

Access the live feed here: [Daily News Website](https://chrischongyj.github.io/daily-news/index)

---

## 🚀 What This Project Does

This project automatically **fetches news from multiple high-quality sources**, including RSS feeds from Bloomberg, Reuters, TechCrunch, Hacker News, and more.  

The workflow is simple but powerful:

1. **Fetch & Extract** – Collects the latest news articles from all configured sources.  
2. **Clean & Deduplicate** – Removes duplicates and extracts the main content from each article.  
3. **AI Analysis** – Uses a Large Language Model (GPT‑4o-mini) to:
   - Summarize articles into concise, readable briefs.
   - Rank each article by importance (impact on investors, founders, or tech professionals).
   - Identify key entities and stock tickers mentioned.  
4. **Publish** – Updates a Markdown feed and optionally converts it into a GitHub Pages website.

---

## 🛠 Technical Complexity

While this project runs seamlessly with a single command, it involves several **advanced techniques**:

- **LangChain & LangGraph** – Orchestrates multi-step pipelines for fetching, analyzing, and publishing news.  
- **LLM Summarization & Classification** – AI automatically understands the significance of articles and condenses them into readable summaries.  
- **Web Scraping & RSS Parsing** – Extracts clean text content from complex web pages.  
- **Database & Versioning** – Uses SQLite to store historical data and GitHub for versioned publishing.  
- **Automation** – Can run daily via cron, automatically updating the website.  

In short, it’s like having a personal AI news analyst, working 24/7 to keep you updated.

---

## ⚡ How to Use

1. Clone the repository.  
2. Set your `OPENAI_API_KEY` environment variable.  
3. Run the pipeline:

```bash
python news_analysis.py --run
