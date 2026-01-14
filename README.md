# 🤖💰 AI Financial Council
<img width="964" height="759" alt="Screenshot 2026-01-05 140416" src="https://github.com/user-attachments/assets/0e7f69d6-40a0-449f-a72e-cbaf9b4c087e" />
<img width="779" height="725" alt="Screenshot 2026-01-05 141148" src="https://github.com/user-attachments/assets/aad8d765-b6b9-4765-8edb-fc2776c40bfe" />

**AI Financial Council** is a sophisticated multi-agent system designed to act as your personal financial advisory board. Built using **LangGraph** and **Google Gemini 2.0 Flash**, it combines personal finance auditing with professional stock market analysis.

This project features a **Streamlit** frontend that provides a chat interface with real-time visibility into the agents' reasoning processes.

## 🧠 System Architecture

This system utilizes a **Supervisor Multi-Agent Architecture**. Instead of a single AI trying to do everything, a central "Supervisor" agent breaks down user queries and delegates tasks to specialized worker agents.

### The Agents (ReAct Architecture)

Each worker agent operates on the **ReAct (Reason + Act)** framework. They don't just answer questions; they iteratively reason about the problem, select the appropriate tools, execute them, and analyze the results before responding.

1. **👮 Supervisor:**
* **Role:** The orchestrator. It analyzes the user's intent and routes the task to the correct specialist or terminates the interaction if the answer is complete.
* **Logic:** Enforces default constraints (e.g., Default High Risk, 60% Capital Allocation) unless specified otherwise.


2. **📈 Stock Analyst Agent:**
* **Role:** Specialized in public market data.
* **Methodology:** Uses a "Three-Pillar" approach:
* *Fundamentals:* Analyzes growth, margins, and valuation via `yfinance`.
* *Sentiment:* Scrapes and summarizes the top 5 news articles using `Newspaper3k`.
* *Technicals:* Calculates RSI, MACD, and Trends.




3. **🧮 Accountant Agent:**
* **Role:** Specialized in personal finance auditing.
* **Capabilities:** securely loads your personal CSV data and writes/executes **Python/Pandas** scripts dynamically to answer questions about spending habits, savings rates, and budget allocation.



### 💾 Memory Systems

The system employs a dual-memory approach to maintain context and personalization:

* **Short-Term Memory (Thread-based):**
* Uses a `checkpointer` to persist the conversation state within a specific session.
* **Auto-Summarization:** To manage context windows efficiently, a dedicated `Summarizer` node activates when the conversation exceeds a certain length, condensing older history while retaining key financial figures.


* **Long-Term Memory (Store-based):**
* Persists user-specific data across different conversations.
* Automatically extracts and updates user profiles, specifically **Risk Tolerance** and **Investment Capital Percentage**, so you don't have to repeat your preferences.



---

## 🛠️ Installation & Setup

### Prerequisites

* Python 3.10+
* A Google Gemini API Key
* A LangSmith API Key (for tracing/debugging)

### 1. Clone the Repository

```bash
git clone https://github.com/Fei0716/ai_financial_council.git
cd ai_financial_council

```

### 2. Environment Configuration

The project requires API keys to function.

1. Locate the `.env.example` file in the root directory.
2. Rename it to `.env`:
```bash
mv .env.example .env

```


3. Open `.env` and add your keys:
```ini
GEMINI_API_KEY=your_google_gemini_key_here
LANGSMITH_API_KEY=your_langsmith_key_here

```



### 3. Install Dependencies

Install the required Python packages used in `financial_council_agents.py` and `app.py`:


### 4. Prepare Data

Ensure your personal finance data is placed correctly:

* Place your cleaned CSV file at: `./data/personal_finance_data_cleaned.csv`

---

## 🚀 Usage

To start the application, open your terminal in the project directory and run:

```bash
streamlit run app.py

```

### Example Queries

* **Stock Analysis:** *"Analyze NVDA and tell me if it's a buy."*
* **Personal Finance:** *"How much did I spend on Food last month?"*
* **Hybrid Planning:** *"Based on my savings, create a stock portfolio for me."* (This triggers the Accountant to find your savings, then the Supervisor calculates 60%, and finally routes to the Analyst for recommendations).

---

## 📂 Project Structure

```text
ai_financial_council/
├── app.py                      # Streamlit Frontend (Chat UI)
├── financial_council_agents.py # Backend Logic (Graph, Agents, Tools)
├── financial_council_agents.ipynb # Development & Testing 
├── .env                        # API Keys (Not committed to git)
├── data/
│   └── personal_finance_data_cleaned.csv  # User data
└── README.md

```

## ⚠️ Disclaimer

This tool is for educational and informational purposes only. It does not constitute certified financial advice. The "AI Financial Council" generates responses based on available data and patterns, but it may hallucinate or provide inaccurate metrics. Always verify financial data and consult a professional advisor before making investment decisions.
