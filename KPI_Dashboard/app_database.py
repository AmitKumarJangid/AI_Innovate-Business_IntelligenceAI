import os
import uuid
import time
import json
import re
import pandas as pd
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from sqlalchemy import create_engine, text
import requests
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
dotenv_path = os.path.join(root_dir, '.env')
load_dotenv(dotenv_path)

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY") 
llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0.1)

# --- TELEMETRY CALCULATOR HELPER ---
def calculate_telemetry(start_time, input_text, output_text, model_name="gemini-1.5-flash"):
    latency_sec = round(time.time() - start_time, 3)
    
    input_tokens = max(1, len(input_text) // 4)
    output_tokens = max(1, len(output_text) // 4)
    total_tokens = input_tokens + output_tokens

    input_cost = (input_tokens / 1_000_000) * 0.075
    output_cost = (output_tokens / 1_000_000) * 0.300
    total_cost = round(input_cost + output_cost, 6)

    return {
        "latency_sec": latency_sec,
        "model_calls": 1,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "estimated_cost_usd": f"${total_cost:.6f}"
    }

template = """
You are an advanced Business Intelligence AI acting as an operational advisor for a retail Bakery. 
USER PERSONA: {persona}
ANOMALY REPORT: {json_payload}

RULES:
1. DO NOT perform arithmetic. 
2. **STRICT ATTRIBUTION RULE:** Check the "attribution_index" in the payload. 
   - If confidence is "High" or "Medium", explicitly state the "primary_driver" and use the supporting reviews or news to explain exactly what happened.
   - If confidence is "Low", DO NOT guess or hypothesize. You MUST state: "Data correlation is inconclusive. Further monitoring required before taking corrective action."
3. **PERSONA ADAPTATION RULE:** You MUST radically shift your tone, vocabulary, and strategic focus based on the USER PERSONA:
   - If "Bakery Owner": Focus strictly on bottom-line profitability, CapEx (capital expenditures), brand equity, and high-level systemic SOPs. Use executive financial terminology.
   - If "Head Chef": Focus strictly on ingredient integrity, recipe standards, kitchen workflow, and back-of-house equipment safety. Use culinary and kitchen-management terminology. Do not focus on front-of-house customer service.
   - If "Store Manager": Focus strictly on front-of-house staff training, immediate customer recovery (comping/discounts), shift scheduling, and daily floor operations.
4. FORMAT: Output cleanly formatted HTML.
5. RECOMMENDATION STRUCTURE: Provide 2 specific recommendations as a bulleted list: 
<b>[Driver]</b> &rarr; [Controllable Lever] &rarr; [Action] &rarr; [Expected Impact] &rarr; [Owner]
"""
prompt = PromptTemplate(input_variables=["persona", "json_payload"], template=template)
chain = prompt | llm

DB_URI = os.getenv("DB_URI")
db_engine = create_engine(DB_URI)

@app.route("/api/kpis", methods=["GET", "POST"])
def manage_kpis():
    try:
        if request.method == "GET":
            df = pd.read_sql(text("SELECT name, formula FROM global_kpis"), con=db_engine)
            kpis = df.to_dict(orient="records")
            return jsonify({"status": "success", "kpis": kpis})

        if request.method == "POST":
            data = request.json
            kpis_to_save = data.get('kpis', [])
            
            with db_engine.begin() as conn:
                conn.execute(text("TRUNCATE TABLE global_kpis"))
                for kpi in kpis_to_save:
                    conn.execute(
                        text("INSERT INTO global_kpis (name, formula) VALUES (:name, :formula)"),
                        {"name": kpi['name'], "formula": kpi['formula']}
                    )
            return jsonify({"status": "success", "message": "KPIs saved globally!"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/fetch_live_data', methods=['GET'])
def fetch_live_data():
    try:
        demo_tenant_id = '1a54985c-8f1c-42ba-8d1c-c81ca8da8cfc' 

        query = text("""
        WITH OrderCOGS AS (
            SELECT 
                oi.order_id,
                SUM(oi.qty * COALESCE(i.cost_price, 0)) as order_cogs
            FROM order_items oi
            LEFT JOIN inventory_items i ON oi.item_id = i.id
            WHERE oi.tenant_id = :tenant_id
            GROUP BY oi.order_id
        )
        SELECT 
            DATE(o.created_at) AS "Unified_Date", 
            COUNT(o.id) AS "Orders",
            SUM(o.total_amount) AS "Revenue",
            SUM(o.discount) AS "Discounts_Given",
            COALESCE(SUM(c.order_cogs), 0) AS "COGS",
            SUM(o.total_amount) - COALESCE(SUM(c.order_cogs), 0) AS "Net_Profit"
        FROM orders o
        LEFT JOIN OrderCOGS c ON o.id = c.order_id
        WHERE o.status != 'cancelled'
          AND o.tenant_id = :tenant_id
        GROUP BY DATE(o.created_at)
        ORDER BY "Unified_Date" ASC;
        """)
        
        df = pd.read_sql(query, con=db_engine, params={"tenant_id": demo_tenant_id})
        filepath = "temp_live_data.csv"
        df.to_csv(filepath, index=False)
        columns = df.columns.tolist()
        
        return jsonify({
            "message": f"Successfully synced {len(df)} days of live production data!",
            "columns": columns,
            "filepath": filepath
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/")
def dashboard():
    return render_template("index.html")

def fetch_external_context(anomaly_date, city_name):
    api_key = os.getenv("NEWS_API_KEY")
    target_date = pd.to_datetime(anomaly_date).strftime('%Y-%m-%d')
    search_query = f"{city_name} AND (rain OR strike OR traffic OR protest OR festival)"
    
    url = (
        f"https://newsapi.org/v2/everything?"
        f"q={search_query}&"
        f"from={target_date}&"
        f"to={target_date}&"
        f"sortBy=relevancy&"
        f"apiKey={api_key}"
    )
    
    try:
        response = requests.get(url, timeout=3).json()
        headlines = [article['title'] for article in response.get('articles', [])[:3]]
        return headlines if headlines else [f"No major events in {city_name} on {target_date}."]
    except Exception:
        return ["External context service temporarily unavailable."]

def calculate_attribution_confidence(internal_reviews, external_news):
    internal_weight = 0.0
    external_weight = 0.0
    
    bad_reviews_count = len(internal_reviews) if internal_reviews else 0
    if bad_reviews_count >= 3:
        internal_weight = 0.85
    elif bad_reviews_count > 0:
        internal_weight = 0.50

    severe_keywords = ["strike", "curfew", "waterlogging", "heavy rain", "closure", "flood", "protest", "festival"]
    
    if external_news and "No major" not in external_news[0]:
        has_severe_event = any(any(k in title.lower() for k in severe_keywords) for title in external_news)
        external_weight = 0.75 if has_severe_event else 0.35

    if internal_weight > external_weight:
        primary_driver = "Internal Operations / Quality"
        score = internal_weight
    elif external_weight > internal_weight:
        primary_driver = "External Macro Event"
        score = external_weight
    else:
        primary_driver = "Inconclusive / Mixed Drivers"
        score = 0.40

    return {
        "primary_driver": primary_driver,
        "confidence_score": round(score, 2),
        "confidence_level": "High" if score >= 0.75 else ("Medium" if score >= 0.50 else "Low")
    }

@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    data = request.json
    filepath = data.get('filepath')
    kpis = data.get('kpis', []) 
    persona = data.get('persona', 'Bakery Owner')

    try:
        df = pd.read_csv(filepath)
        df['Unified_Date'] = pd.to_datetime(df['Unified_Date']).dt.strftime('%Y-%m-%d')
        date_col = 'Unified_Date'

        all_results = []

        for kpi in kpis:
            kpi_name = kpi['name']
            kpi_formula = kpi['formula']
            
            df[kpi_name] = df.eval(kpi_formula)
            
            ema_mean = df[kpi_name].ewm(span=14, adjust=False).mean()
            ema_std = df[kpi_name].ewm(span=14, adjust=False).std()
            
            z_scores = (df[kpi_name] - ema_mean) / ema_std
            z_scores = z_scores.fillna(0.0)
            
            valid_idx = df.index
            if len(valid_idx) > 0:
                today_idx = valid_idx[-1] 
                today_z = z_scores[today_idx]
                
                is_anomalous = abs(today_z) >= 1.5
                
                start_idx = max(0, today_idx - 13)
                chart_slice = df.loc[start_idx:today_idx]
                
                formula_vars = [word for word in re.findall(r'[A-Za-z_][A-Za-z0-9_]*', kpi_formula) if word in df.columns]
                driver_impacts_raw = []
                
                if is_anomalous:
                    for var in formula_vars:
                        if today_idx >= 14:
                            baseline = df.loc[today_idx-14 : today_idx-1, var].mean()
                        else:
                            baseline = df.loc[:today_idx, var].mean()
                            
                        anomaly_day_val = df.loc[today_idx, var]
                        
                        if baseline and baseline != 0:
                            pct_change = ((anomaly_day_val - baseline) / baseline) * 100
                            driver_impacts_raw.append({"driver": var, "raw_value": float(pct_change), "formatted": f"{pct_change:+.1f}%"})
                        else:
                            driver_impacts_raw.append({"driver": var, "raw_value": 0.0, "formatted": "N/A"})

                final_ranked_drivers = sorted(driver_impacts_raw, key=lambda x: x['raw_value']) if is_anomalous else []
                
                anomaly_payload = {
                    "kpi_name": kpi_name,
                    "formula_used": kpi_formula,
                    "anomaly_date": df.loc[today_idx, date_col],
                    "z_score": round(float(today_z), 2),
                    "observed_value": round(float(df.loc[today_idx, kpi_name]), 2),
                    "top_detractor": final_ranked_drivers[0]['driver'] if final_ranked_drivers else "None",
                    "ranked_drivers": final_ranked_drivers,
                    "chart_labels": chart_slice[date_col].tolist(),
                    "chart_data": chart_slice[kpi_name].round(2).tolist()
                }
                
                if is_anomalous:
                    start_time = time.time()
                    demo_tenant_id = '1a54985c-8f1c-42ba-8d1c-c81ca8da8cfc' 
                    reviews_query = text("""
                        SELECT item_name, rating, review_text 
                        FROM product_reviews 
                        WHERE tenant_id = :tenant_id 
                          AND rating <= 3 
                        ORDER BY created_at DESC 
                        LIMIT 5;
                    """)
                    city_query = text("SELECT city FROM tenants WHERE id = :tenant_id")
                    
                    with db_engine.connect() as conn:
                        reviews_result = conn.execute(reviews_query, {"tenant_id": demo_tenant_id})
                        recent_bad_reviews = [dict(row) for row in reviews_result.mappings()]
                        tenant_city = conn.execute(city_query, {"tenant_id": demo_tenant_id}).scalar() or "Bhopal"
                    
                    today_str = str(df.loc[today_idx, 'Unified_Date'])[:10]
                    anomaly_payload["recent_negative_customer_feedback"] = recent_bad_reviews
                    anomaly_payload["external_news_context"] = fetch_external_context(today_str, tenant_city)

                    confidence_data = calculate_attribution_confidence(recent_bad_reviews, anomaly_payload["external_news_context"])
                    anomaly_payload["attribution_index"] = confidence_data

                    prompt_payload = json.dumps(anomaly_payload, indent=2)
                    response = chain.invoke({"persona": persona, "json_payload": prompt_payload})
                    ai_report = response.content[0].get('text', '') if isinstance(response.content, list) else response.content
                    
                    telemetry = calculate_telemetry(start_time, prompt_payload, ai_report)
                    anomaly_payload["ai_report"] = ai_report
                    anomaly_payload["telemetry"] = telemetry
                else:
                    anomaly_payload["ai_report"] = """
                    <div class='alert alert-success mt-4'>
                        <h5 class='alert-heading mb-2'>Status: Normal 🟢</h5>
                        <p class='mb-0'>This metric is operating within normal statistical parameters. No action required.</p>
                    </div>
                    """
                    anomaly_payload["telemetry"] = {
                        "latency_sec": 0,
                        "model_calls": 0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                        "estimated_cost_usd": "$0.000000"
                    }
                    
                all_results.append(anomaly_payload)

        all_results = sorted(all_results, key=lambda x: abs(x['z_score']), reverse=True)

        return jsonify({
            "status": "success",
            "anomalies": all_results
        })

    except Exception as e:
        return jsonify({"error": f"Math Engine Error: {str(e)}"}), 500

# --- UPDATED CHATBOT ROUTE: CONCISE INTENT + STATISTICAL LAYER + LIVE NEWS ---
@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.json
    user_query = data.get("query", "").strip()
    filepath = data.get("filepath", "temp_live_data.csv")
    persona = data.get("persona", "Bakery Owner")

    if not os.path.exists(filepath):
        return jsonify({
            "status": "error",
            "reply": "No active dataset found. Please click 'Fetch Latest Production Data' in Step 1 first."
        })

    try:
        start_time = time.time()
        df = pd.read_csv(filepath)
        columns = df.columns.tolist()

        # 1. DATA-GAP INTERCEPTOR
        common_missing = ["inventory", "waste", "footfall", "labor", "weather", "supplier", "margin"]
        query_words = [re.sub(r'\W+', '', w).lower() for w in user_query.split()]
        
        missing_detected = []
        for word in query_words:
            if word in common_missing and word not in [c.lower() for c in columns]:
                missing_detected.append(word)

        if missing_detected:
            reply_text = f"Data Gap Warning: Column(s) {', '.join(missing_detected)} missing."
            telemetry = calculate_telemetry(start_time, user_query, reply_text)
            telemetry["model_calls"] = 0
            return jsonify({
                "status": "success",
                "reply": f"""
                <div class='alert alert-warning border-0 shadow-sm mb-0'>
                    <strong>⚠️ Data Gap Detected:</strong><br>
                    The bot cannot compute calculations for <code>{', '.join(missing_detected)}</code> because these columns are missing from the current active dataset.<br><br>
                    <strong>Available Columns in Engine:</strong> {', '.join([f'<code>{c}</code>' for c in columns])}
                </div>
                """,
                "telemetry": telemetry
            })

        # 2. STATISTICAL ENGINE LAYER (Computed mathematically before calling LLM)
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        stat_summary = {}
        for col in numeric_cols:
            mean_val = float(df[col].mean())
            std_val = float(df[col].std())
            latest_val = float(df[col].iloc[-1])
            z_score = float((latest_val - mean_val) / std_val) if std_val != 0 else 0.0
            stat_summary[col] = {
                "latest_value": round(latest_val, 2),
                "mean": round(mean_val, 2),
                "std_dev": round(std_val, 2),
                "latest_z_score": round(z_score, 2),
                "is_anomalous": abs(z_score) >= 1.5
            }

        # 3. NEWS API INTEGRATION (Triggered if macro event context requested)
        news_data = []
        if any(k in user_query.lower() for k in ["news", "event", "external", "weather", "rain", "strike"]):
            anomaly_date = str(df['Unified_Date'].iloc[-1]) if 'Unified_Date' in df.columns else "2026-03-01"
            demo_tenant_id = '1a54985c-8f1c-42ba-8d1c-c81ca8da8cfc'
            with db_engine.connect() as conn:
                tenant_city = conn.execute(text("SELECT city FROM tenants WHERE id = :tenant_id"), {"tenant_id": demo_tenant_id}).scalar() or "Bhopal"
            news_data = fetch_external_context(anomaly_date, tenant_city)

        # 4. DYNAMIC GRAPH HANDLER
        is_graph_request = any(k in user_query.lower() for k in ["graph", "plot", "chart", "draw", "vs", "trend", "compare"])
        chart_data = None

        if is_graph_request:
            requested_cols = [c for c in numeric_cols if c.lower() in user_query.lower()]
            if len(requested_cols) >= 2:
                x_col, y_col = requested_cols[0], requested_cols[1]
            elif len(requested_cols) == 1:
                x_col = 'Unified_Date' if 'Unified_Date' in df.columns else numeric_cols[0]
                y_col = requested_cols[0]
            else:
                x_col = 'Unified_Date' if 'Unified_Date' in df.columns else numeric_cols[0]
                y_col = numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0]

            chart_data = {
                "chart_id": f"chat_chart_{int(time.time())}",
                "x_col": x_col,
                "y_col": y_col,
                "labels": df[x_col].astype(str).tolist(),
                "values": df[y_col].round(2).tolist(),
                "anomalies_count": int(stat_summary.get(y_col, {}).get("is_anomalous", False))
            }

        # 5. CONCISE CONVERSATIONAL PROMPT (Strict output formatting rules)
        chat_prompt = f"""
        You are an AI Business Intelligence Assistant.
        USER PERSONA: {persona}
        ENGINE COMPUTED STATISTICAL SUMMARY: {json.dumps(stat_summary, indent=2)}
        AVAILABLE COLUMNS: {columns}
        NEWS CONTEXT: {news_data if news_data else "None requested"}

        USER QUERY: "{user_query}"

        STRICT RESPONSE LAWS:
        1. STICK ONLY TO WHAT THE USER ASKED. If the user asks a simple question (e.g., "what can you do", "hi", "what columns exist"), respond concisely in 2-3 sentences max.
        2. DO NOT include detailed operational summaries, anomaly deep-dives, or structured recommendations UNLESS the user explicitly asks for "recommendations", "deep analysis", or "insights".
        3. If the user asks for data analysis, ground every claim in the ENGINE COMPUTED STATISTICAL SUMMARY provided above (mention exact values, mean, or z-scores).
        4. Output cleanly formatted HTML without markdown code blocks.
        """

        bot_response = llm.invoke([HumanMessage(content=chat_prompt)])
        reply_html = bot_response.content[0].get('text', '') if isinstance(bot_response.content, list) else bot_response.content

        telemetry = calculate_telemetry(start_time, chat_prompt, reply_html)

        return jsonify({
            "status": "success",
            "reply": reply_html,
            "chart": chart_data,
            "telemetry": telemetry
        })

    except Exception as e:
        return jsonify({"status": "error", "reply": f"Chatbot Error: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)
