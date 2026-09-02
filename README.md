# DecisivAI — Autonomous KPI Storytelling & Decision Intelligence Engine

**DecisivAI** is an autonomous decision intelligence platform submitted by **Team AI_Innovates** for the **Accenture Innovation Challenge 2026**. It bridges raw transactional metrics with actionable strategic execution by combining deterministic statistical anomaly detection, multi-grain data reconciliation, external news and review scraping, and persona-driven AI storytelling.

The platform features a **dual-layer reasoning architecture**: a deterministic statistical pre-filtering layer in Python first validates real signals ($Z$-scores, EMA baselines, waterfall driver math) and calculates a Unified Confidence Index. An LLM layer (`gemini-3.6-flash`) is then called only when valid anomalies occur, generating evidence-backed root cause reports and structured recommendation chains[cite: 1, 4].

---

## Core Capabilities & Technical Highlights

* **Dual-Mode Data Ingestion:** 
  * **Live Production Database Sync:** Ingests live transactional records directly from PostgreSQL / Supabase databases via SQLAlchemy, executing multi-table relational joins across orders, order items, inventory COGS, and tenant profiles.
  * **Heterogeneous File Uploads:** Ingests and reconciles primary and secondary CSV or Excel files across different collection cadences (e.g., daily POS orders vs. weekly ad spend) via automatic date alignment (`Unified_Date`) and forward/backward filling.
* **Deterministic Anomaly & Waterfall Engine:**
  * Calculates rolling Exponential Moving Averages (EMA, 14-period span) and dynamic standard deviations to compute absolute $Z$-scores ($\vert{}Z\vert{} \ge 1.5$).
  * Automatically isolates formula variables and computes percentage driver contributions against 14-day historical baselines to rank top detractors.
* **Contextual Retrieval & News Scraper (RAG):**
  * Fetches real-time external macro context (news events, weather, traffic, local disruptions) targeted strictly to the anomaly date and merchant city using the News API.
  * Retrieves internal unstructured customer feedback (product reviews $\le 3$ stars) directly from Supabase.
* **Attribution Confidence Index & Guardrails:**
  * Computes an explicit **Attribution Confidence Score** (High $\ge 0.75$, Medium $\ge 0.50$, Low $< 0.50$) based on review density and macro event severity.
  * Enforces **Abstention Rules**: If data history is insufficient ($<14$ periods) or attribution confidence is Low, the system halts or explicitly refuses to hypothesize, preventing AI hallucinations[cite: 1, 4].
* **Dynamic Persona Adaptation:** 
  * Tailors output tone, strategic scope, and vocabulary across distinct leadership roles[cite: 4]:
    * **Bakery Owner / C-Suite:** Focuses on bottom-line profitability, CapEx, brand equity, and systemic SOPs.
    * **Head Chef / Operations:** Focuses on ingredient integrity, recipe standards, kitchen workflow, and back-of-house equipment safety.
    * **Store Manager:** Focuses on front-of-house staffing, immediate customer recovery, shift scheduling, and daily floor execution.
* **Global KPI Persistence:** Saves and manages user-defined formulas centrally in the global database (`global_kpis` table).

---

## System Architecture


                               ┌────────────────────────────────────────┐
                               │       Multi-Source Data Ingestion      │
                               └───────────────────┬────────────────────┘
                                                   │
                         ┌─────────────────────────┴────────────────────────┐
                         ▼                                                  ▼
           [ Live Supabase / PostgreSQL DB ]                       [ Heterogeneous File Upload ]
         (SQL Joins: Orders, Items, Inventory)                   (CSV / XLSX Merging & ffill/bfill)
                         └─────────────────────────┬────────────────────────┘
                                                   │
                                                   ▼
                                  ┌──────────────────────────────────┐
                                  │ Deterministic Engine (Math Layer) │
                                  └────────────────┬─────────────────┘
                                                   │
                                                   ├── EMA Rolling Baseline (14-period)
                                                   ├── Z-Score Threshold Check (|Z| >= 1.5)
                                                   └── Waterfall Driver Attribution Math
                                                   │
                                                   ▼
                                  ┌──────────────────────────────────┐
                                  │      Abstention Gatekeeper       │
                                  └────────────────┬─────────────────┘
                                                   │
                         ┌─────────────────────────┴────────────────────────┐
                         │                                                  │
       (Insufficient History < 14 Days)                        (Valid Anomaly Detected)
                         │                                                  │
                         ▼                                                  ▼
            [ ABSTENTION GUARDRAIL ]                       [ Contextual RAG Retrieval ]
          (Halt LLM to stop hallucination)              ├── Internal Low-Rating Customer Reviews
                                                        └── External News API (City + Date Scrape)
                                                                            │
                                                                            ▼
                                                           [ Attribution Confidence Scoring ]
                                                        (Calculates High / Medium / Low Score)
                                                                            │
                                                                            ▼
                                                           [ LangChain + Gemini 3.6 Flash ]
                                                        (Persona-Based HTML Report Generation)
                                                                            │
                                                                            ▼
                                                           [ Dashboard / Action Delivery ]


## Technical Stack

- **Backend Framework:** Python Flask
- **Database & ORM:** PostgreSQL / Supabase, SQLAlchemy, PyMySQL / `psycopg2`
- **Data Processing:** Pandas, NumPy, Werkzeug
- **Generative AI & Orchestration:** LangChain, `langchain-google-genai` (`gemini-3.6-flash`)
- **External APIs:** News API (Real-time news search)
- **Configuration:** `python-dotenv`

## Installation & Setup Guide

### 1. Prerequisites

- Python 3.9+
- Active Supabase / PostgreSQL Database instance
- Google Gemini API Key
- News API Key

### 2. Repository Setup

```Bash


git clone [https://github.com/your-username/decisiv-ai.git](https://github.com/your-username/decisiv-ai.git)
cd decisiv-ai
```

### 3. Environment Isolation

```Bash


python -m venv venv
```

# Windows:
venv\Scripts\activate

# macOS / Linux:
source venv/bin/activate

### 4. Install Dependencies

```Bash


pip install flask pandas werkzeug sqlalchemy requests langchain-google-genai langchain-core python-dotenv openpyxl
```

### 5. Environment Configuration (`.env`)

Create a `.env` file in the root directory:

```

GOOGLE_API_KEY=your_gemini_api_key_here
NEWS_API_KEY=your_news_api_key_here
DB_URI=postgresql://user:password@db.supabase.co:5432/postgres
```

### 6. Database Schema Setup (Supabase / PostgreSQL)

Ensure your database contains the required baseline tables for live production sync:
For example, the tables we used are:

SQL

```sql 
CREATE TABLE global_kpis (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    formula VARCHAR(255) NOT NULL
);

CREATE TABLE tenants (
    id UUID PRIMARY KEY,
    name VARCHAR(255),
    city VARCHAR(255)
);

CREATE TABLE orders (
    id UUID PRIMARY KEY,
    tenant_id UUID REFERENCES tenants(id),
    total_amount NUMERIC,
    discount NUMERIC,
    status VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE inventory_items (
    id UUID PRIMARY KEY,
    tenant_id UUID REFERENCES tenants(id),
    cost_price NUMERIC
);

CREATE TABLE order_items (
    id UUID PRIMARY KEY,
    order_id UUID REFERENCES orders(id),
    item_id UUID REFERENCES inventory_items(id),
    tenant_id UUID REFERENCES tenants(id),
    qty INT
);

CREATE TABLE product_reviews (
    id UUID PRIMARY KEY,
    tenant_id UUID REFERENCES tenants(id),
    item_name VARCHAR(255),
    rating INT,
    review_text TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

```

## Running the Application

### Option A: Running the Supabase Live Production Engine

To run the server connected to live database tables, external news API scraping, and persona adaptation:

```Bash


python app_supabase.py
```

### Option B: Running the CSV/XLSX File Merger Engine

To run the server configured for local multi-file upload and offline analysis:

```Bash


python app_csv.py
```

## API Documentation

### 1. Live Data Sync (`GET /api/fetch_live_data`)

- **Description:** Connects to Supabase, executes SQL CTE queries combining `orders`, `order_items`, and `inventory_items`, calculates daily revenue, COGS, discounts, and net profits, and dumps the synced dataset for mathematical processing.

### 2. Manage Global KPIs (`GET` / `POST /api/kpis`)

- **GET:** Returns all active global KPI definitions from Supabase (`global_kpis`).
- **POST:** Truncates and updates global KPI formulas dynamically across tenant instances.

### 3. Upload & Reconcile Files (`POST /api/upload`)

- **Description:** Merges primary and secondary heterogeneous files on `Unified_Date`, applying forward and backward filling to sync lower-cadence metrics[cite: 4].

### 4. Execute Analysis & Storytelling (`POST /api/analyze`)

- **Description:** Evaluates formulas, calculates EMA rolling baselines, runs Z-score anomaly checks, retrieves reviews and news context, assesses attribution confidence, and renders persona-adapted HTML action cards.

- **Payload Example:**
JSON
 ```json
 

  
  {
    "filepath": "temp_live_data.csv",
    "persona": "Bakery Owner",
    "kpis": [
      {
        "name": "Net_Profit_Margin",
        "formula": "(Net_Profit / Revenue) * 100"
      }
    ]
  }
```

### 4. Execute Analysis & Storytelling (`POST /api/analyze`)

- **Description:** Evaluates formulas, calculates EMA rolling baselines, runs Z-score anomaly checks, retrieves reviews and news context, assesses attribution confidence, and renders persona-adapted HTML action cards.

- **Payload Example:**

  JSON

  ```json
  {
    "filepath": "temp_live_data.csv",
    "persona": "Bakery Owner",
    "kpis": [
      {
        "name": "Net_Profit_Margin",
        "formula": "(Net_Profit / Revenue) * 100"
      }
    ]
  }


### 5. Team Composition

- **Team Name:** AI_Innovates
- **Competition:** Accenture Innovation Challenge 2026
- **Product Name:** DecivAI
