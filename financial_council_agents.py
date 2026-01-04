import operator
import os
import io
import contextlib
import pandas as pd
from datetime import datetime
from typing import Annotated, Sequence, TypedDict, List, Union, Literal, Optional
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage, RemoveMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
import yfinance as yf
from newspaper import Article
from datetime import datetime
from dotenv import load_dotenv

# --- CONFIGURATION ---
load_dotenv("./.env")  # Load .env file

#setup LangSmith for tracing and debugging
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Financial_Council_Agent"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")


#states for accountant agent
ACTIVE_DATAFRAME = None
CSV_FILE_PATH = "./data/personal_finance_data_cleaned.csv"

# Gemini 2.0 Flash
llm_flash = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=os.getenv("GEMINI_API_KEY"), temperature=0)

# setup long term and short term memory
# Short-Term Memory (Persists conversation within a thread)
checkpointer = InMemorySaver()
config = {
    "configurable": {
        "thread_id": "13",    #Change this to start a new conversation
        "user_id": "user_123"
    }
}

# Long-Term Memory (Persists user facts across threads)
store = InMemoryStore()


### HELPER FUNCTIONS DECLARATIONS
def summarize_text(text, ticker):
    """
    Helper to summarize raw news article text using  llm_flash instance.
    """
    if not text or len(text) < 200:
        return text  # text is too short to summarize

    try:
        # Define the prompt instructions
        system_prompt = (
            "You are a financial news summarizer. Summarize the following article in 3 bullet points. "
            "Focus specifically on how it impacts the stock price or future outlook. Be concise."
        )
        user_content = f"Ticker: {ticker}\nArticle Text: {text[:4000]}"  # Truncate to save tokens

        # Create the message payload for LangChain
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content)
        ]

        # Invoke Gemini Flash
        response = llm_flash.invoke(messages)

        # Return the actual text content from the AIMessage
        return response.content

    except Exception as e:
        return f"Summarization failed: {e}. Raw snippet: {text[:200]}..."


def update_profile_memory(state, config, store):
    """
    Updates the Long-Term Memory.
    """
    # 1. Setup
    user_id = config["configurable"]["user_id"]
    namespace = ("user", user_id)
    key = "financial_profile"

    # 2. Get Existing Memory (or default)
    existing_item = store.get(namespace, key)
    current_profile = existing_item.value if existing_item else {}

    # 3. Define Extraction Prompt
    # We ask the LLM to look at the LAST few messages (where the new info likely is)
    system_msg = """
    You are a Memory Manager. 
    Analyze the conversation to extract User Preferences.
    - Risk Tolerance (Low/Med/High)
    - Preferred Investment Allocation % (e.g., "I want to invest 50% of my savings")

    CRITICAL:
    - DO NOT extract specific dollar amounts (e.g., Net Savings, Income). 
    - We must recalculate those fresh every time. Only remember the *rules* (percentages), not the *values*.
    """

    # 4. Invoke LLM with Structured Output
    # We only look at the last 5 messages to save tokens and focus on recent updates
    recent_messages = state['messages'][-5:]

    extractor = llm_flash.with_structured_output(UserProfile)
    extracted_data = extractor.invoke([SystemMessage(content=system_msg)] + recent_messages)

    # 5. Merge / Patch Logic
    # We only update keys that are NOT None in the extraction
    has_updates = False
    updated_profile = current_profile.copy()

    data_dict = extracted_data.model_dump(exclude_none=True)
    for k, v in data_dict.items():
        # Update if value is different
        if updated_profile.get(k) != v:
            updated_profile[k] = v
            has_updates = True

    # 6. Save back to Store
    if has_updates:
        store.put(namespace, key, updated_profile)
        # Optional: Print for debugging
        print(f"💾 MEMORY UPDATED: {data_dict}")

    return updated_profile


def filter_agent_context(messages, current_agent_name):
    """
    Filters and TRANSFORMS message history to prevent confusion.

    1. System/User/Supervisor -> Keep as is.
    2. Current Agent (Me) -> Keep as AIMessage (My memories).
    3. Other Agents -> Convert to HumanMessage with "[Agent Name]:" prefix (External input).
    """
    filtered = []
    valid_tool_call_ids = set()

    for msg in messages:
        # --- 1. User / System / Supervisor ---
        if isinstance(msg, (HumanMessage, SystemMessage)) or msg.name == "Supervisor":
            filtered.append(msg)
            continue

        # --- 2. AI Messages ---
        if isinstance(msg, AIMessage):
            # CASE A: It's ME (Keep it as 'Assistant')
            if msg.name == current_agent_name:
                filtered.append(msg)
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tc in msg.tool_calls:
                        valid_tool_call_ids.add(tc['id'])

            # CASE B: It's ANOTHER Agent (Transform to 'User/Human' Input)
            else:
                if msg.content and str(msg.content).strip():
                    # We pretend this is a User telling the agent what happened.
                    # This creates a clear boundary: "Accountant said X" vs "I said X"
                    transcribed_msg = HumanMessage(
                        content=f"[{msg.name}]: {msg.content}",
                        name=msg.name,
                        id=msg.id
                    )
                    filtered.append(transcribed_msg)

        # --- 3. Tool Messages ---
        if isinstance(msg, ToolMessage):
            # Only keep tool results if *I* asked for them
            if msg.tool_call_id in valid_tool_call_ids:
                filtered.append(msg)

    return filtered


def extract_new_messages(full_history, input_messages_len, agent_name):
    """
    Extracts only the new messages generated by the agent, preventing duplicates.

    Why this is critical:
    1. 'create_react_agent' returns the full history (Input + New).
    2. If we injected a Summary SystemMessage, the index shifts.
    3. Sometimes internal chains echo HumanMessages.

    This function slices correctly and strictly filters for AI outputs only.
    """
    # 1. Slice based on the EXACT length of the input used for this run
    # This removes the old history and any temporary injected context (Summary)
    potential_new_messages = full_history[input_messages_len:]

    clean_messages = []

    for msg in potential_new_messages:
        # 2. We discard HumanMessage or SystemMessage to prevent echoing user input.
        if isinstance(msg, (AIMessage, ToolMessage)):
            # 3. Stamp the Agent's Name, this ensures the Supervisor knows exactly who said what.
            msg.name = agent_name
            clean_messages.append(msg)

    return clean_messages


# --- DEFINE STATE ---
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    summary: str
    next_step: str

# Define the valid routing choices
class SupervisorDecision(BaseModel):
    next_step: Literal["StockAnalyst", "Accountant", "FINISH"] = Field(
        description="The next agent to act or FINISH if the task is complete."
    )
    instructions: str = Field(
        description="Specific instructions for the selected agent. If FINISH, must provide a final summary for the user."
    )

class TickerInput(BaseModel):
    ticker: str = Field(..., description="The stock ticker symbol (e.g., 'NVDA', 'AAPL') to analyze.")

class FilePathInput(BaseModel):
    file_path: str = Field(description="Path to the CSV file to load. Defaults to internal path if not provided.",
                           default=CSV_FILE_PATH)

class ColumnInput(BaseModel):
    column_name: str = Field(..., description="The column to check for unique values (e.g., 'Category').")

class PythonScriptInput(BaseModel):
    script: str = Field(...,
                        description="Executable Python Pandas code. The dataframe is named 'df'. The final result must be printed.")

class UserProfile(BaseModel):
    """
    Long-term memory that stores user preferences.
    """
    risk_tolerance: Optional[Literal["Low", "Medium", "High"]] = Field(
        None, description="User's risk tolerance. Infer 'High' if they ask for aggressive stocks."
    )
    investment_capital_percent: Optional[float] = Field(
        None, description="The percentage of savings the user prefers to invest (default is usually 60%)."
    )


### TOOLS DECLARATIONS
# for stock analyst agent
@tool(args_schema=TickerInput)
def get_stock_fundamentals(ticker: str):
    """
    Retrieves key financial metrics including 3-Year historical growth,
    Next 2-Year forecasted growth, Margins, FCF, and Valuation Ratios.
    """
    try:
        stock = yf.Ticker(ticker)

        # 1. Fetch Key Dataframes
        info = stock.info
        financials = stock.financials  # Income Statement
        cashflow = stock.cashflow  # Cash Flow Statement
        estimates = stock.revenue_estimate  # Revenue Estimation by Analyst
        q_financials = stock.quarterly_financials  # Quarterly Financial Data
        q_cashflow = stock.quarterly_cashflow  # Cash Flow Statement

        # --- PREPARATION: Get Basic Variables ---
        market_cap = stock.fast_info['market_cap']

        if market_cap is None:
            # Fallback if fast_info fails (rare)
            market_cap = info.get('marketCap')

        # Extract the last 4 quarters (columns 0 to 4) to get the TTM Data
        # yfinance columns are usually sorted by date descending (newest first)
        last_4_quarters_financials = q_financials.iloc[:, :4]
        last_4_quarters_cash_flow = q_cashflow.iloc[:, :4]
        # Sum them to get TTM values
        ttm_op_income = last_4_quarters_financials.loc['Operating Income'].sum()
        ttm_revenue = last_4_quarters_financials.loc['Total Revenue'].sum()
        ttm_fcf = last_4_quarters_cash_flow.loc['Free Cash Flow'].sum()

        # Get Revenue (Current vs 3-Year Ago)
        try:
            rev_current = financials.loc['Total Revenue'].iloc[0]
            rev_3yr_ago = financials.loc['Total Revenue'].iloc[3]
        except (IndexError, KeyError):
            rev_current = None
            rev_3yr_ago = None

        # Get Free Cash Flow (Current vs 3-Year Ago)
        try:
            # Year 0 (Current)
            ocf_curr = cashflow.loc['Operating Cash Flow'].iloc[0]
            capex_curr = cashflow.loc['Capital Expenditure'].iloc[0]
            fcf_current = ocf_curr + capex_curr

            # Year 3 (3 Years Ago)
            ocf_3yr = cashflow.loc['Operating Cash Flow'].iloc[3]
            capex_3yr = cashflow.loc['Capital Expenditure'].iloc[3]
            fcf_3yr_ago = ocf_3yr + capex_3yr
        except (IndexError, KeyError):
            fcf_current = None
            fcf_3yr_ago = None

        # --- METRIC 1: Past 3-Year Revenue Growth (CAGR) ---
        if rev_current and rev_3yr_ago and rev_3yr_ago > 0:
            growth_3y = ((rev_current / rev_3yr_ago) ** (1 / 3)) - 1
        else:
            growth_3y = None

        # --- METRIC 2: Next 2 Years Forecasted Revenue Growth ---
        # Formula: ((Avg Est Next Year / Year Ago Sales Current Year) ^ 0.5) - 1
        forecast_growth = None
        if estimates is None or estimates.empty:
            return "N/A"
        try:
            # 1. Get the Numerator: Average Estimated Revenue for Next Year (+1y)
            # We look for the '+1y' row and the 'avg' column
            next_year_est = estimates.loc['+1y', 'avg']

            # 2. Get the Denominator: Year-ago Revenue for Current Year (0y)
            # We look for the '0y' row and the 'yearAgoRevenue' column
            # This represents the actual confirmed revenue from the last completed fiscal year.
            last_year_actual = estimates.loc['0y', 'yearAgoRevenue']

            # 3. Apply the Formula: ((Next Year Est / Last Year Actual) ^ 0.5) - 1
            forecast_growth = ((next_year_est / last_year_actual) ** 0.5) - 1

        except Exception as e:
            print(e)

        # --- METRIC 3: Operating Margin ---
        try:
            # Calculate Margin
            op_margin = ttm_op_income / ttm_revenue

        except KeyError as e:
            print(f"N/A (Key not found: {e})")
        except IndexError:
            print("N/A (Not enough quarterly data available)")
        except Exception as e:
            print(f"N/A (Error: {e})")

        # --- METRIC 4: FCF Growth / Change ---
        fcf_metric_name = "FCF Growth (3y)"
        fcf_metric_value = None

        if fcf_current is not None and fcf_3yr_ago is not None:
            if fcf_current > 0 and fcf_3yr_ago > 0:
                # CAGR Formula
                fcf_metric_value = ((fcf_current / fcf_3yr_ago) ** (1 / 3)) - 1
            else:
                # Absolute Change Formula (handles negative FCF)
                fcf_metric_name = "FCF Change (3y)"
                fcf_metric_value = (fcf_current - fcf_3yr_ago) / abs(fcf_3yr_ago)

        # --- METRIC 5: Valuation Ratios ---
        ps_ratio = market_cap / ttm_revenue if (market_cap and ttm_revenue) else None
        pfcf_ratio = market_cap / ttm_fcf if (market_cap and fcf_current) else None

        # --- FORMAT OUTPUT ---
        def fmt_pct(val):
            return f"{val:.2%}" if val is not None else "N/A"

        def fmt_num(val):
            return f"${val:,.0f}" if val is not None else "N/A"

        def fmt_rat(val):
            return f"{val:.2f}" if val is not None else "N/A"

        return {
            "Ticker": ticker.upper(),
            "Current Market Cap": fmt_num(market_cap),
            "3-Year Revenue Growth": fmt_pct(growth_3y),
            "Next 2-Year Forecast Growth": fmt_pct(forecast_growth) if forecast_growth else "N/A (No Analyst Data)",
            "Operating Margin": fmt_pct(op_margin),
            f"{fcf_metric_name}": fmt_pct(fcf_metric_value),
            "Latest FCF": fmt_num(fcf_current),
            "Price to Sales (P/S)": fmt_rat(ps_ratio),
            "Price to FCF (P/FCF)": fmt_rat(pfcf_ratio)
        }

    except Exception as e:
        return f"Failed to analyze {ticker}: {str(e)}"


@tool(args_schema=TickerInput)
def get_stock_news(ticker: str):
    """
    Fetches top 5 latest news items for a stock ticker, scrapes news content, AND summarizes it using llm_flash.
    """
    try:
        # 1. Get Links
        # Using a try-except block specifically for yf.Search to be safe
        try:
            search = yf.Search(ticker, news_count=5)
            news_list = search.news
        except Exception:
            news_list = []

        # Fallback to Ticker if Search returns nothing (redundancy)
        if not news_list:
            try:
                t = yf.Ticker(ticker)
                news_list = t.news
            except Exception:
                pass

        if not news_list:
            return "No news found."

        formatted_news = []
        print(f"Found {len(news_list)} links. Scraping & Summarizing with Gemini Flash...")

        for item in news_list:
            link = item.get('link')

            # 2. Extract Content
            article_content = ""
            try:
                if link:
                    article = Article(link)
                    article.download()
                    article.parse()
                    article_content = article.text
            except Exception:
                # If scraping fails, just skip this article or mark as failed
                continue

                # 3. Summarize Content
            if article_content:
                # Call our helper function that uses llm_flash
                summary = summarize_text(article_content, ticker)
            else:
                summary = "N/A (Could not extract text)"

            # 4. Format Output
            pub_time = item.get('providerPublishTime')
            date_str = datetime.fromtimestamp(pub_time).strftime('%Y-%m-%d') if pub_time else "N/A"

            formatted_news.append({
                "title": item.get('title', 'No Title'),
                "date": date_str,
                "publisher": item.get('publisher', 'Unknown'),
                "link": link,
                "summary": summary
            })

        return formatted_news

    except Exception as e:
        return f"Error in news tool: {e}"


@tool(args_schema=TickerInput)
def get_technical_indicators(ticker: str) -> dict:
    """
    Calculates key technical indicators (RSI, MACD, SMA) for a given stock.
    Useful for determining price trends and potential entry/exit points.
    """
    try:
        t = yf.Ticker(ticker)
        # Fetch 6 months of data to ensure enough history for 200-day or 50-day calc
        # Note: We need at least 26+9=35 days for MACD, so 6mo is safe.
        hist = t.history(period="6mo")

        if hist.empty:
            return {"error": "No historical price data found."}

        # --- A. Simple Moving Averages (SMA) ---
        # SMA 50: Short-term trend
        hist['SMA_50'] = hist['Close'].rolling(window=50).mean()

        # --- B. Relative Strength Index (RSI) ---
        # Formula: 100 - (100 / (1 + RS))
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()

        rs = gain / loss
        hist['RSI'] = 100 - (100 / (1 + rs))

        # --- C. MACD (Moving Average Convergence Divergence) ---
        # EMA 12 - EMA 26
        ema_12 = hist['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = hist['Close'].ewm(span=26, adjust=False).mean()
        hist['MACD'] = ema_12 - ema_26
        # Signal Line: 9-day EMA of MACD
        hist['MACD_Signal'] = hist['MACD'].ewm(span=9, adjust=False).mean()

        # --- Get Latest Values ---
        latest = hist.iloc[-1]

        # Determine Trend (Simple Logic for Agent)
        price = latest['Close']
        sma_50 = latest['SMA_50']
        trend = "Uptrend" if price > sma_50 else "Downtrend"

        # Determine RSI Status
        rsi = latest['RSI']
        rsi_status = "Neutral"
        if rsi > 70:
            rsi_status = "Overbought (High Risk of Pullback)"
        elif rsi < 30:
            rsi_status = "Oversold (Potential Bounce)"

        return {
            "current_price": f"${price:.2f}",
            "sma_50": f"${sma_50:.2f}" if pd.notnull(sma_50) else "N/A",
            "trend_50_day": trend,
            "rsi_14": f"{rsi:.1f}",
            "rsi_status": rsi_status,
            "macd_line": f"{latest['MACD']:.2f}",
            "macd_signal": f"{latest['MACD_Signal']:.2f}",
            "macd_interpretation": "Bullish Cross" if latest['MACD'] > latest['MACD_Signal'] else "Bearish"
        }

    except Exception as e:
        return {"error": f"Technical analysis failed: {str(e)}"}


# for accountant agent
@tool(args_schema=FilePathInput)
def read_financial_dataset(file_path: str = CSV_FILE_PATH) -> str:
    """
    Loads the personal finance CSV into memory and returns its structure (columns, types, first 3 rows).
    Use this FIRST to understand the data.
    """
    global ACTIVE_DATAFRAME
    try:
        df = pd.read_csv(file_path)
        # Clean headers: lowercase and strip spaces for easier coding
        df.columns = [c.strip() for c in df.columns]
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

        ACTIVE_DATAFRAME = df  # Store in memory

        buffer = []
        buffer.append(f"Columns: {list(df.columns)}")
        buffer.append(f"Data Types: {df.dtypes.to_dict()}")
        buffer.append(f"Shape: {df.shape} (Rows, Columns)")
        buffer.append("\nSample Data (First 3 rows):")
        buffer.append(df.head(3).to_markdown(index=False))

        return "\n".join(buffer)
    except Exception as e:
        return f"Error loading CSV: {e}"


@tool(args_schema=ColumnInput)
def get_unique_values(column_name: str) -> str:
    """
    Returns the top 50 unique values from a specific column.
    Useful for knowing exact category names (e.g., 'Food & Drink' vs 'Dining') before filtering.
    """
    global ACTIVE_DATAFRAME
    if ACTIVE_DATAFRAME is None:
        return "Error: No data loaded. Call read_financial_dataset first."

    try:
        # Check case-insensitive
        col_map = {c.lower(): c for c in ACTIVE_DATAFRAME.columns}
        if column_name.lower() in col_map:
            actual_col = col_map[column_name.lower()]
            uniques = ACTIVE_DATAFRAME[actual_col].unique().tolist()
            return f"Unique values in '{actual_col}' ({len(uniques)} total):\n{uniques[:50]}"
        else:
            return f"Error: Column '{column_name}' not found. Available: {list(ACTIVE_DATAFRAME.columns)}"
    except Exception as e:
        return f"Error: {e}"


@tool(args_schema=PythonScriptInput)
def execute_pandas_script(script: str) -> str:
    """
    Executes a python script on the loaded 'df'.
    IMPORTANT: You must PRINT() the final result to see it.
    Example: print(df[df['Amount'] > 0]['Amount'].sum())
    """
    global ACTIVE_DATAFRAME
    if ACTIVE_DATAFRAME is None:
        return "Error: No data loaded. Call read_financial_dataset first."

    # Create a safe capture buffer for stdout
    output_buffer = io.StringIO()

    try:
        # Define the local environment with the dataframe
        local_env = {"df": ACTIVE_DATAFRAME, "pd": pd}

        # Redirect stdout to capture print() statements
        with contextlib.redirect_stdout(output_buffer):
            exec(script, {}, local_env)

        result = output_buffer.getvalue()
        if not result:
            return "Code executed successfully but printed nothing. Did you forget `print(...)`?"
        return result

    except Exception as e:
        return f"Python Execution Error: {e}"


# --- DEFINE NODES ---
supervisor_system_prompt = """You are the Supervisor of a financial team.
You have two workers:
1. StockAnalyst: Specialized in public market data, stock prices, and investment metrics.
2. Accountant: Specialized in personal finance, reading the user's local CSV expenses, incomes, and budgeting.

**DEFAULT ASSUMPTIONS (CRITICAL):**
If the user does not provide specific constraints, you **MUST** assume the following. **DO NOT ask the user for these details:**
- **Risk Tolerance:** DEFAULT:HIGH (Aggressive Growth).
- **Investment Capital:** DEFAULT:60% of the user's Total Net Savings (Income - Expenses).

**SCOPE OF ACTION (CRITICAL):**
- **Focus ONLY on the LAST User message.** Do not answer or summarize questions from previous turns that have already been resolved.
- If the User asks "What about Intel?", your final answer should ONLY talk about Intel. **DO NOT** bring up previous expense reports or travel budgets unless explicitly asked to compare them.

**WORKFLOW PROTOCOL:**
1. **ALLOCATION & BUDGET REQUESTS (Requires Accountant):**
   - **Trigger:** ONLY if the user explicitly asks **'How much should I invest?'**, **'Create a portfolio'**, **'Allocate my funds'**, or **'Can I afford X?'**.
   - **Step 1:** Route to **Accountant**. Instruction: 'Calculate Total Net Savings (Total Income - Total Expenses) and the current Monthly Savings Rate.'
   - **Step 2:** Upon receiving the savings figure, calculate 60% of that value yourself.
   - **Step 3:** Route to **StockAnalyst**. Instruction: 'Recommend stocks for a HIGH RISK portfolio with an investment budget of $[INSERT_60%_VALUE].'

2. **PURE ANALYSIS REQUESTS (Direct Routing):**
   - **Trigger:** User asks 'Analyze NVDA', 'Is Tesla a buy?', 'What are good AI stocks?', 'How is the market?'.
   - **Action:** Route DIRECTLY to **StockAnalyst**. **DO NOT** call the Accountant. **DO NOT** mention budget.

3. **PERSONAL FINANCE REQUESTS:**
   - **Trigger:** User asks about spending, savings, or income.
   - **Action:** Route DIRECTLY to **Accountant**.

4. **FINISHING:**
   - **Condition:** If the sub-agent has returned an answer that satisfies the **LATEST** User request.
   - **Action:** Select 'FINISH'.
   - **Instructions:** "Present the answer to the user. **STRICTLY** omit information related to previous, resolved queries."
"""

stock_analyst_system_prompt = """
You are a Senior Investment Analyst for the 'AI Financial Council'.
Your goal is to evaluate public companies using a holistic "Three-Pillar" approach: Fundamentals, Sentiment, and Technicals.
You MUST use the three tools provided to you to do those three analysis. 
You have one co-worker:
1. Accountant: Specialized in personal finance, reading the user's local CSV expenses, incomes, and budgeting

Instructions:
1.  **Competitor Discovery:** If the user asks for analysis of a stock and its "competitors" (e.g., "NVDA and 4 AI competitors") without providing tickers, **DO NOT ask the user.** Autonomously identify the top publicly traded competitors and proceed.

2.  **Tool Use Strategy:**
    - **Fundamentals (`get_stock_fundamentals`):** Step 1. Assess the *health* of the business (Growth, Margins, Valuation).
    - **Sentiment (`get_stock_news`):** Step 2. Check the *story* (Catalysts, Risks, Market Mood).
    - **Technicals (`get_technical_indicators`):** Step 3. Check the *timing* (Trends, RSI, Momentum).
    - **Mandate:** For comprehensive requests (e.g., "Analyze NVDA"), you must use **ALL THREE** tools to provide a complete recommendation.

3.  **Synthesis & Recommendations:**
    - **The "Buy" Case:** Strong Fundamentals + Positive News + Bullish Technicals (or Oversold RSI).
    - **The "Wait" Case:** Strong Fundamentals but Overbought RSI (>70) or Negative News.
    - **The "Avoid" Case:** Declining Fundamentals (Negative Growth/FCF).

4.  **Metric Focus:**
    - **Fundamentals:** High Growth (>20%), High Margins (>20%), and Positive FCF are key. P/S > 15 implies premium valuation.
    - **Technicals:** RSI < 30 is Oversold (Potential Buy), > 70 is Overbought (Risk). MACD Crossovers signal momentum shifts.

5.  **Output Format:**
    - **Executive Summary:** One clear sentence recommendation (Buy, Hold, or Watch).
    - **Deep Dive:** Distinct sections for Fundamentals, News/Sentiment, and Technical Analysis.
    - **Risks:** What could go wrong?

6.  **Investment Allocation & Budgeting (CRITICAL LOGIC):**
    - **STEP 1: Check User Intent.** Did the user explicitly ask "How much should I buy?", "What is the allocation?", or "Build me a portfolio"?

      - **NO (Pure Analysis Request):** - Provide the 3-pillar analysis and your Buy/Hold/Sell rating. 
        - **STOP.** Do NOT mention the Accountant. Do NOT ask for a budget.

      - **YES (Allocation Request):** - **Check Instructions:** Did the Supervisor provide a specific budget (e.g., "$50,000")?
        - **Scenario A (Budget Provided):** Recommend specific dollar amounts based on your conviction.
        - **Scenario B (Budget Missing):** State: *"I have analyzed the stocks, but I cannot recommend specific investment amounts without knowing your financial standing. Supervisor, please request the Accountant to analyze the user's available capital."*
"""

accountant_system_prompt = """
You are an expert Accountant Agent responsible for personal financial auditing.
You MUST use the three tools provided to you. 
You have one co-worker:
1. StockAnalyst: Specialized in public market data, stock prices, and investment metrics.

Your goal is to answer user questions by analyzing their transaction data programmatically using Python and Pandas.
**Context:** The current year is **2024** (or match user's context), but the dataset may contain historical data.

### **CORE WORKFLOW**
1.  **Ingestion:** ALWAYS start by running `read_financial_dataset` to load the data and see the columns.
2.  **Pre-processing (CRITICAL):** The 'Date' column often loads as a string. In your Python scripts, **ALWAYS** convert it first if analyzing time:
    ```python
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    ```
3.  **Validation:** Before filtering by text (e.g., "Food"), must use `get_unique_values` to confirm the exact spelling (e.g., is it "Food & Drink" or "Restaurants"?).

4.  **Execution:** Write Python code using `execute_pandas_script`.
    - Variable `df` is pre-loaded.
    - **CRITICAL SYNTAX RULE:** When writing the python script, **USE SINGLE QUOTES (')** for all internal strings (e.g., `df['Category']`). Do NOT use double quotes inside the script, as this breaks the JSON formatting.
    - You MUST `print()` the final result to see it.

### **ANALYTICAL CAPABILITIES**
-   **Spending Breakdown:** Group expenses by Category to find the biggest money leaks.
    * *Code Hint:* `df[df['Type']=='Expense'].groupby('Category')['Amount'].sum().sort_values(ascending=False)`
-   **Trend Analysis:** Resample data by Month to see if spending is rising.
    * *Code Hint:* `df.set_index('Date').resample('ME')['Amount'].sum()`
-   **Savings Rate:** Always calculate (Total Income - Total Expense) / Total Income.
-   **Outlier Detection:** Find the single largest transaction in a given period.

### **CONSTRAINTS**
-   **Never guess numbers.** If the code returns an empty result, say "No data found for those criteria."
-   **Privacy:** Do not print the entire dataset. Print only summaries or top 5 rows.
-   **Error Handling:** If a script fails (e.g., "KeyError"), correct the column name using the metadata from Step 1 and try again.
-   **Financial data source:** Do not need to ask users for the path of their finance data, as `read_financial_dataset` tool has access to it already.
"""

stock_agent_runnable = create_react_agent(
    llm_flash,
    [get_stock_fundamentals, get_stock_news, get_technical_indicators],
    prompt=stock_analyst_system_prompt,
    # debug=True,
)
accountant_runnable = create_react_agent(
    llm_flash,
    [read_financial_dataset, get_unique_values, execute_pandas_script],
    prompt=accountant_system_prompt
)


def supervisor_node(state):
    messages = state['messages']
    summary = state.get('summary', "")

    previous_summary = ""
    if summary:
        previous_summary += f"**PREVIOUS CONVERSATION SUMMARY:**\n{summary}\n\n"

    # Update Memory (Extract Risk/Preferences only)
    user_profile = update_profile_memory(state, config, store)

    # Inject Memory Context
    # We now ONLY inject preferences, so the Supervisor doesn't hallucinate a budget
    memory_context = "**USER PREFERENCES (LONG-TERM):**\n"
    if user_profile.get('risk_tolerance'):
        memory_context += f"- Risk Tolerance: {user_profile['risk_tolerance']}\n"
    if user_profile.get('investment_capital_percent'):
        memory_context += f"- Preferred Allocation Rule: {user_profile['investment_capital_percent']:.0%} of Net Savings\n"

    # We concatenate the memory context at the top of the system prompt
    final_system_prompt = previous_summary + "\n" + memory_context + "\n" + supervisor_system_prompt

    dynamic_route_prompt = ChatPromptTemplate.from_messages([
        ("system", final_system_prompt),
        ("placeholder", "{messages}"),
    ])

    # Create the router chain 'with_structured_output' ensures we get a clean JSON/Object back, not a chatty sentence
    supervisor_chain = dynamic_route_prompt | llm_flash.with_structured_output(SupervisorDecision)
    decision = supervisor_chain.invoke({"messages": messages})

    # Handle the 'FINISH' case (Stop the loop)
    if decision.next_step == "FINISH":
        # add the final summary to the last message
        final_message = AIMessage(
            content=decision.instructions,
            name="Supervisor"
        )
        return {
            "next_step": "FINISH",
            "messages": [final_message]
        }

    # Handle Delegation (Pass Instruction)
    # We create a new message that looks like it came from the Supervisor to the Worker
    # This overrides the original user query for the sub-agent's context
    instruction_message = HumanMessage(
        content=f"[SUPERVISOR INSTRUCTION]: {decision.instructions}",
        name="Supervisor"
    )

    # Return the step AND the new instruction message to be appended to history
    return {
        "next_step": decision.next_step,
        "messages": [instruction_message]
    }


def stock_analyst_node(state):
    """
    Invokes the prebuilt ReAct agent.
    The agent will loop internally until it generates a final text response.
    """
    all_messages = state['messages']
    summary = state.get("summary", "")

    # analyst to focus on the user's goal and the Supervisor's instruction
    filtered_messages = filter_agent_context(all_messages, "StockAnalyst")

    if summary:
        summary_msg = SystemMessage(
            content=f"**PREVIOUS CONVERSATION CONTEXT:**\n{summary}"
        )
        invocation_messages = [summary_msg] + filtered_messages
    else:
        invocation_messages = filtered_messages

    # Invoke with CLEAN context
    result = stock_agent_runnable.invoke({"messages": invocation_messages})

    # Extract new messages
    new_messages = extract_new_messages(
        full_history=result['messages'],
        input_messages_len=len(invocation_messages),
        agent_name="StockAnalyst"
    )

    return {"messages": new_messages}


def accountant_node(state):
    """
    Invokes the Accountant ReAct agent.
    """
    all_messages = state['messages']
    summary = state.get("summary", "")

    # analyst to focus on the user's goal and the Supervisor's instruction
    filtered_messages = filter_agent_context(all_messages, "Accountant")

    if summary:
        summary_msg = SystemMessage(
            content=f"**PREVIOUS CONVERSATION CONTEXT:**\n{summary}"
        )
        invocation_messages = [summary_msg] + filtered_messages
    else:
        invocation_messages = filtered_messages

    result = accountant_runnable.invoke({"messages": invocation_messages})

    # Extract new messages
    new_messages = extract_new_messages(
        full_history=result['messages'],
        input_messages_len=len(invocation_messages),
        agent_name="Accountant"
    )

    return {"messages": new_messages}


def summarizer_node(state):
    messages = state['messages']
    current_summary = state.get('summary', "")

    # 1. Safety Check
    if len(messages) < 6:
        return {"messages": []}

    # 2. Slice Logic
    to_summarize = messages[:-4]

    # 3. Convert to Text (Append to existing summary if needed)
    conversation_text = ""
    for msg in to_summarize:
        if isinstance(msg, RemoveMessage): continue

        # We handle the previous summary differently now (it's in 'state', not 'messages')
        # So we only summarize the *new* conversation messages here.
        role = msg.name if msg.name else msg.type
        if msg.content:
            conversation_text += f"{role}: {msg.content}\n"

    # 4. Generate New Summary
    # We ask the LLM to merge the OLD summary with the NEW conversation
    prompt = (
        "Update the running conversation summary with the new lines of dialogue.\n"
        "Keep it concise and retain financial figures.\n\n"
        f"OLD SUMMARY:\n{current_summary}\n\n"
        f"NEW CONVERSATION:\n{conversation_text}"
    )

    new_summary_response = llm_flash.invoke(prompt)

    # 5. Delete Old Messages
    delete_messages = [RemoveMessage(id=m.id) for m in to_summarize]

    # 6. RETURN UPDATE
    # We update the 'summary' key and clean up the 'messages' list
    return {
        "summary": new_summary_response.content,  # Updates the State field
        "messages": delete_messages  # Cleans the Chat History
    }


def should_summarize(state):
    messages = state['messages']
    if len(messages) > 15:
        return "summarize"
    return "continue"

# --- BUILD GRAPH ---
workflow = StateGraph(AgentState)
workflow.add_node("Supervisor", supervisor_node)
workflow.add_node("StockAnalyst", stock_analyst_node)
workflow.add_node("Accountant", accountant_node)
workflow.add_node("Summarizer", summarizer_node)

# Check length first
def entry_point(state):
    if len(state['messages']) > 15:
        return "Summarizer"
    return "Supervisor"

workflow.set_conditional_entry_point(
    entry_point,
    {"Summarizer": "Summarizer", "Supervisor": "Supervisor"}
)
workflow.add_edge("Summarizer", "Supervisor")
workflow.add_conditional_edges("Supervisor", lambda x: x['next_step'],
    {"StockAnalyst": "StockAnalyst", "Accountant": "Accountant", "FINISH": END})
workflow.add_edge("StockAnalyst", "Supervisor")
workflow.add_edge("Accountant", "Supervisor")

# COMPILE WITH MEMORY
app = workflow.compile(
    checkpointer=checkpointer,
    store=store
)