# 📰 Agentic AI News App
An Agentic AI-powered news aggregator that searches the web and fetches news from multiple APIs, then summarizes and ranks top articles with the power of LLMs + embeddings + vector storage.

Built for curious minds who want real-time, reliable, and contextual news insights. 🔍✨

## 🚀 Features
🌍 Multi-source Aggregation – Fetches top stories from news APIs and real-time web search.

🤖 AI-powered Answers – Query any topic and get curated summaries with source links.

🔎 Smart Search – Use embeddings + Tavily web search for precise results.

📝 Summarized Context – Understand trending news in seconds via Gemma-8B LLM.

🗂 Vector-powered Memory – Stores embeddings in ChromaDB for efficient retrieval.

## 🛠️ Tech Stack
LLM: Gemma-8B 🧠

Embeddings: MiniLM (for semantic search with Tavily)

Vector DB: ChromaDB 📦

Retriever Tool: Tavily 🌐

Orchestration: Agentic AI workflows (tools + reasoning)

## 🔄 Workflow
1️⃣ User enters a topic or question.
2️⃣ Agent triggers:

📡 News APIs → Get trusted sources.

🌐 Tavily Web Search → Fetch real-time coverage.
3️⃣ Store in ChromaDB with MiniLM embeddings.
4️⃣ Gemma-8B LLM → Summarize + rank results.
5️⃣ 🎯 Output a clean list of articles + insights.

## 💡 Example Use Cases
🔬 “AI advancements in 2025” → Curated list of top-rated articles.

## ⚙️ Installation
Clone the repository:

bash
git clone https://github.com/your-username/agentic-ai-news-app.git
cd agentic-ai-news-app
Install dependencies:

bash
pip install -r requirements.txt
▶️ Usage
Run the app:

bash
python app.py
Search for a topic:

bash
Enter a topic: "Climate Change"
Ask a contextual query:

## 🔥 With Agentic AI News App, stay ahead of the curve with smarter, faster, and more reliable news insights.
