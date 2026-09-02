import os
import uuid
import time
import json
import re
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

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

template = """
You are an advanced Business Intelligence AI acting as an operational advisor. 
USER PERSONA: {persona}
ALL KPI ANOMALY REPORTS: {json_payload}

RULES:
1. DO NOT perform arithmetic or fabricate numbers.
2. **ATTRIBUTION & CONFIDENCE DISPLAY RULE:** Check the "attribution_index" in the payload:
   - If confidence_level is "High" or "Medium": State the "primary_driver" and cite the supporting customer reviews to explain the root cause.
   - If confidence_level is "Low": State that data correlation is inconclusive due to limited feedback density, and explicitly state the confidence score and level:
     "<p class='text-warning'><b>Data correlation is inconclusive (Confidence Score: [confidence_score], Level: Low).</b> Insufficient review density to determine a definitive root cause.</p>"
3. **RECOMMENDATION RULE (HYPOTHETICAL VS EVIDENCE-BASED):**
   - If confidence_level is "High" or "Medium": Provide 2-3 targeted recommendations directly tied to the review evidence.
   - If confidence_level is "Low": Provide 2-3 **hypothetical recommendations** based on plausible operational risks. Clearly state that these recommendations are hypothetical owing to the low confidence score ([confidence_score] / Low).
4. **REVIEW DIRECTIONALITY RULE:**
   - For KPI Drop anomalies: Explain drivers using customer reviews with ratings <= 3.
   - For KPI Rise anomalies: Explain drivers using customer reviews with ratings >= 4.
5. **PERSONA ADAPTATION RULE:** Adapt your tone, vocabulary, and operational focus based on the USER PERSONA:
   - "Financial Analyst": Focus on margins, CapEx, cost variance, revenue impact, and financial ROI.
   - "CEO": Focus on strategic growth, brand equity, operational scaling, and revenue.
   - "Store Manager": Focus on daily floor execution, customer satisfaction, staffing shift optimization, and inventory levels.
   - "Head Chef" / "Inventory Operations": Focus on raw material quality, workflow efficiency, scrap/waste control, and safety standard operating procedures.
6. FORMAT: Output cleanly formatted HTML.
RECOMMENDATION STRUCTURE:
Provide 2-3 recommendations as a bulleted list:
<b>[Driver / Potential Driver]</b> &rarr; [Controllable Lever] &rarr; [Action] &rarr; [Expected Impact] &rarr; [Owner]
"""
prompt = PromptTemplate(input_variables=["persona", "json_payload"], template=template)
chain = prompt | llm

@app.route("/")
def dashboard():
    return render_template("index.html")


@app.route("/api/upload", methods=["POST"])
def api_upload():
    if 'file1' not in request.files:
        return jsonify({"error": "Primary metrics dataset file is required."}), 400
    
    try:
        f1 = request.files['file1']
        df1 = pd.read_csv(f1) if f1.filename.endswith('.csv') else pd.read_excel(f1)
        
        # Standardize Primary File Date Column
        date_col_1 = df1.columns[0]
        df1[date_col_1] = pd.to_datetime(df1[date_col_1], errors='coerce')
        df1 = df1.rename(columns={date_col_1: 'Unified_Date'})
        df1 = df1.sort_values('Unified_Date')

        if df1.empty:
            return jsonify({"error": "Uploaded primary dataset is empty."}), 400

        actual_unique_dates = df1['Unified_Date'].nunique()
        if actual_unique_dates < 14:
            return jsonify({
                "error": f"Guardrail Triggered: Dataset contains only {actual_unique_dates} distinct date(s). At least 14 unique dates are required."
            }), 400

        filename_primary = secure_filename(f"{uuid.uuid4().hex}_primary.csv")
        filepath_primary = os.path.join(app.config['UPLOAD_FOLDER'], filename_primary)
        df1.to_csv(filepath_primary, index=False)

        # Process Optional Secondary CSV File (Customer Reviews)
        reviews_filepath = None
        df2 = None
        if 'file2' in request.files and request.files['file2'].filename != '':
            f2 = request.files['file2']
            filename_reviews = secure_filename(f"{uuid.uuid4().hex}_reviews.csv")
            reviews_filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename_reviews)
            f2.save(reviews_filepath)
            
            try:
                df2 = pd.read_csv(reviews_filepath)
                date_col_2 = None
                for col in df2.columns:
                    if 'date' in col.lower():
                        date_col_2 = col
                        break
                if date_col_2:
                    df2['Unified_Date'] = pd.to_datetime(df2[date_col_2], dayfirst=True, errors='coerce').dt.strftime('%Y-%m-%d')
            except Exception as e:
                print(f"Error parsing secondary file during upload: {e}")

        # Extract Combined Numeric & Categorical Columns from Primary AND Secondary Sources
        numeric_columns = []
        categorical_columns = []
        
        # 1. Primary Source Columns
        for c in df1.columns:
            if c == 'Unified_Date':
                continue
            if pd.api.types.is_numeric_dtype(df1[c]):
                if c not in numeric_columns:
                    numeric_columns.append(c)
            else:
                if c not in categorical_columns:
                    categorical_columns.append(c)

        # 2. Secondary Source Columns
        if df2 is not None:
            excluded_text_cols = ['Unified_Date', 'Date', 'Review_ID', 'Customer_ID', 'Review_Text']
            for c in df2.columns:
                if c in excluded_text_cols:
                    continue
                if pd.api.types.is_numeric_dtype(df2[c]):
                    if c not in numeric_columns:
                        numeric_columns.append(c)
                else:
                    if c not in categorical_columns:
                        categorical_columns.append(c)

        return jsonify({
            "filepath": filepath_primary, 
            "reviews_filepath": reviews_filepath,
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def fetch_csv_reviews(reviews_filepath, anomaly_date, anomaly_type="drop", segment_dim=None, segment_val=None, limit=5):
    if not reviews_filepath or not os.path.exists(reviews_filepath):
        return []
    
    try:
        df_rev = pd.read_csv(reviews_filepath)
        
        date_col = None
        for col in df_rev.columns:
            if 'date' in col.lower():
                date_col = col
                break
                
        if date_col:
            df_rev['Parsed_Date'] = pd.to_datetime(df_rev[date_col], dayfirst=True, errors='coerce').dt.strftime('%Y-%m-%d')
        
        # Rating Filter based on Anomaly Type
        if 'Rating' in df_rev.columns:
            if anomaly_type == "drop":
                df_filtered = df_rev[df_rev['Rating'] <= 3]
            else:
                df_filtered = df_rev[df_rev['Rating'] >= 4]
        else:
            df_filtered = df_rev.copy()
            
        # Segment Filter
        if segment_dim and segment_val and segment_dim in df_filtered.columns:
            df_filtered = df_filtered[df_filtered[segment_dim].astype(str).str.lower() == str(segment_val).lower()]
            
        # Date Matching
        if date_col and anomaly_date:
            target_dt = str(anomaly_date)[:10]
            date_matches = df_filtered[df_filtered['Parsed_Date'] == target_dt]
            if len(date_matches) > 0:
                df_filtered = date_matches
            else:
                try:
                    target_dt_obj = pd.to_datetime(target_dt)
                    start_dt_obj = target_dt_obj - pd.Timedelta(days=7)
                    parsed_dts = pd.to_datetime(df_filtered['Parsed_Date'], errors='coerce')
                    df_filtered = df_filtered[(parsed_dts >= start_dt_obj) & (parsed_dts <= target_dt_obj)]
                except Exception:
                    pass

        cols_to_grab = ['Review_Text', 'Rating', 'Category', 'Region', 'Product_ID', 'Sentiment']
        existing_cols = [c for c in cols_to_grab if c in df_filtered.columns]
        
        reviews_list = []
        for _, row in df_filtered.head(limit).iterrows():
            item = {c: (float(row[c]) if isinstance(row[c], (int, float)) else str(row[c])) for c in existing_cols}
            reviews_list.append(item)
            
        return reviews_list

    except Exception as e:
        print(f"Error parsing secondary reviews CSV: {e}")
        return []


def calculate_attribution_confidence(reviews, anomaly_type="drop"):
    reviews_count = len(reviews) if reviews else 0
    
    if reviews_count >= 3:
        score = 0.85
        confidence_level = "High"
        primary_driver = "Customer Feedback / Quality Issues" if anomaly_type == "drop" else "Customer Satisfaction / High Demand"
    elif reviews_count >= 1:
        score = 0.50
        confidence_level = "Medium"
        primary_driver = "Customer Feedback / Quality Issues" if anomaly_type == "drop" else "Customer Satisfaction / High Demand"
    else:
        score = 0.30
        confidence_level = "Low"
        primary_driver = "Inconclusive / Insufficient Feedback"

    return {
        "primary_driver": primary_driver,
        "confidence_score": round(score, 2),
        "confidence_level": confidence_level
    }


def calculate_kpi_anomalies(df_input, kpis, reviews_filepath=None, segment_dim=None, segment_val=None, date_col='Unified_Date'):
    processed_kpis = []
    
    for kpi in kpis:
        kpi_name = kpi['name']
        kpi_formula = kpi['formula']
        
        df = df_input.copy()
        try:
            df[kpi_name] = df.eval(kpi_formula)
        except Exception:
            continue
        
        # 14-Day Baseline Calculations
        rolling_mean = df[kpi_name].rolling(window=14).mean()
        rolling_std = df[kpi_name].rolling(window=14).std()
        z_scores = (df[kpi_name] - rolling_mean) / rolling_std
        z_scores = z_scores.fillna(0.0)
        
        recent_indices = df.tail(14).index
        valid_idx = [i for i in recent_indices if i in z_scores.index]
        
        if len(valid_idx) > 0:
            abs_z_scores = z_scores.loc[valid_idx].abs()
            local_max_abs_idx = abs_z_scores.idxmax()
            local_z_score = z_scores.loc[local_max_abs_idx]
            anomaly_type = "drop" if local_z_score < 0 else "rise"
            
            formula_vars = [word for word in re.findall(r'[A-Za-z_][A-Za-z0-9_]*', kpi_formula) if word in df.columns]
            driver_impacts_raw = []
            
            for var in formula_vars:
                if local_max_abs_idx >= 14:
                    baseline = df.loc[local_max_abs_idx-14 : local_max_abs_idx-1, var].mean()
                else:
                    baseline = df.loc[:local_max_abs_idx, var].mean()
                    
                anomaly_day_val = df.loc[local_max_abs_idx, var]
                
                if baseline and baseline != 0:
                    pct_change = ((anomaly_day_val - baseline) / baseline) * 100
                    driver_impacts_raw.append({"driver": var, "raw_value": float(pct_change), "formatted": f"{pct_change:+.1f}%"})
                else:
                    driver_impacts_raw.append({"driver": var, "raw_value": 0.0, "formatted": "N/A"})

            ranked_drivers = sorted(driver_impacts_raw, key=lambda x: abs(x['raw_value']), reverse=True)
            chart_slice = df.tail(14)
            anomaly_date_str = str(df.loc[local_max_abs_idx, date_col])[:10]

            # Fetch contextual customer reviews
            relevant_reviews = fetch_csv_reviews(
                reviews_filepath=reviews_filepath, 
                anomaly_date=anomaly_date_str, 
                anomaly_type=anomaly_type,
                segment_dim=segment_dim,
                segment_val=segment_val
            )

            # Compute Confidence Index
            attribution_data = calculate_attribution_confidence(relevant_reviews, anomaly_type)

            processed_kpis.append({
                "kpi_name": kpi_name,
                "formula_used": kpi_formula,
                "anomaly_date": anomaly_date_str,
                "z_score": round(float(local_z_score), 2),
                "abs_z_score": round(float(abs(local_z_score)), 2),
                "anomaly_type": anomaly_type,
                "observed_value": round(float(df.loc[local_max_abs_idx, kpi_name]), 2),
                "top_detractor": ranked_drivers[0]['driver'] if ranked_drivers else "None",
                "ranked_drivers": ranked_drivers,
                "chart_labels": chart_slice[date_col].astype(str).tolist(),
                "chart_data": chart_slice[kpi_name].round(2).tolist(),
                "customer_reviews": relevant_reviews,
                "attribution_index": attribution_data
            })
            
    processed_kpis.sort(key=lambda x: x['abs_z_score'], reverse=True)
    return processed_kpis


def create_aggregated_dataset(df_raw):
    """
    Applies 'mean' to rates/scores (e.g., Rating) and 'sum' to financial/volume metrics.
    """
    agg_dict = {}
    numeric_cols = df_raw.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in ['rating', 'score', 'margin', 'pct', 'rate', 'avg']):
            agg_dict[col] = 'mean'
        else:
            agg_dict[col] = 'sum'
            
    return df_raw.groupby('Unified_Date').agg(agg_dict).reset_index().sort_values('Unified_Date')


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    data = request.json
    filepath = data.get('filepath')
    reviews_filepath = data.get('reviews_filepath')
    kpis = data.get('kpis', []) 
    persona = data.get('persona', 'Financial Analyst')
    group_by_cols = data.get('group_by_cols', [])

    try:
        # Load Primary Source Dataset
        df_primary = pd.read_csv(filepath)
        df_primary['Unified_Date'] = pd.to_datetime(df_primary['Unified_Date']).dt.strftime('%Y-%m-%d')

        # Load & Align Secondary Source Dataset if uploaded
        if reviews_filepath and os.path.exists(reviews_filepath):
            df_secondary = pd.read_csv(reviews_filepath)
            date_col_2 = None
            for col in df_secondary.columns:
                if 'date' in col.lower():
                    date_col_2 = col
                    break
            if date_col_2:
                df_secondary['Unified_Date'] = pd.to_datetime(df_secondary[date_col_2], dayfirst=True, errors='coerce').dt.strftime('%Y-%m-%d')
            
            # Find common join key dimensions (e.g., Unified_Date, Category, Region)
            common_keys = [c for c in ['Unified_Date', 'Category', 'Region', 'Store_ID', 'Product_ID'] if c in df_primary.columns and c in df_secondary.columns]
            if not common_keys:
                common_keys = ['Unified_Date']

            df_raw = pd.merge(df_primary, df_secondary, on=common_keys, how='left')
        else:
            df_raw = df_primary

        if len(df_raw['Unified_Date'].unique()) < 14:
            return jsonify({
                "ai_report": "<p class='text-danger'><b>Guardrail Triggered:</b> Insufficient date history.</p>",
                "overall_kpis": [],
                "segment_anomalies": {},
                "telemetry": {"latency": "0.0s"}
            })

        # Overall Aggregation across combined primary and secondary source columns
        df_overall = create_aggregated_dataset(df_raw)
        overall_kpis = calculate_kpi_anomalies(df_overall, kpis, reviews_filepath=reviews_filepath)

        # Segment Aggregations
        segment_anomalies = {}
        ANOMALY_Z_THRESHOLD = 1.5

        for dim_col in group_by_cols:
            if dim_col in df_raw.columns:
                unique_segments = df_raw[dim_col].dropna().unique()
                flagged_segments = []

                for seg_val in unique_segments:
                    df_seg = df_raw[df_raw[dim_col] == seg_val]
                    df_seg_agg = create_aggregated_dataset(df_seg)
                    
                    if len(df_seg_agg) >= 14:
                        seg_kpi_results = calculate_kpi_anomalies(
                            df_seg_agg, kpis, 
                            reviews_filepath=reviews_filepath,
                            segment_dim=dim_col,
                            segment_val=seg_val
                        )
                        if seg_kpi_results:
                            max_z = seg_kpi_results[0]['abs_z_score']
                            if max_z >= ANOMALY_Z_THRESHOLD:
                                flagged_segments.append({
                                    "segment_name": str(seg_val),
                                    "dimension": dim_col,
                                    "max_z_score": seg_kpi_results[0]['z_score'],
                                    "abs_z_score": max_z,
                                    "anomaly_type": seg_kpi_results[0]['anomaly_type'],
                                    "kpis": seg_kpi_results
                                })

                if flagged_segments:
                    flagged_segments.sort(key=lambda x: x['abs_z_score'], reverse=True)
                    segment_anomalies[dim_col] = flagged_segments

        # Payload sent to LLM
        llm_payload = {
            "overall_kpis": overall_kpis,
            "segment_anomalies": {dim: [s['segment_name'] for s in segs] for dim, segs in segment_anomalies.items()}
        }

        start_time = time.time()
        response = chain.invoke({"persona": persona, "json_payload": json.dumps(llm_payload, indent=2)})
        ai_report = response.content[0].get('text', '') if isinstance(response.content, list) else response.content
        latency = round(time.time() - start_time, 2)

        return jsonify({
            "ai_report": ai_report,
            "overall_kpis": overall_kpis,
            "segment_anomalies": segment_anomalies,
            "telemetry": {"latency": f"{latency}s", "model": "gemini-3.6-flash"}
        })

    except Exception as e:
        return jsonify({"error": f"Math Engine Error: {str(e)}"}), 500


@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.json or {}
    filepath = data.get('filepath')
    reviews_filepath = data.get('reviews_filepath')
    user_message = data.get('message', '').strip()

    if not filepath or not os.path.exists(filepath):
        return jsonify({"reply": "<p class='text-danger'>No dataset loaded. Please complete Step 1 and run analysis first.</p>", "chart": None})

    try:
        # Load Datasets into Context
        df_primary = pd.read_csv(filepath)
        
        context_parts = [
            f"--- PRIMARY DATASET METRICS ---",
            f"Total Rows: {len(df_primary)}",
            f"Columns: {list(df_primary.columns)}",
            f"Sample Data (First 10 rows):\n{df_primary.head(10).to_string(index=False)}"
        ]

        if df_primary.select_dtypes(include=np.number).columns.any():
            context_parts.append(f"Primary Numerical Summary:\n{df_primary.describe().to_string()}")

        if reviews_filepath and os.path.exists(reviews_filepath):
            df_secondary = pd.read_csv(reviews_filepath)
            context_parts.extend([
                f"\n--- SECONDARY CUSTOMER REVIEWS DATASET ---",
                f"Total Rows: {len(df_secondary)}",
                f"Columns: {list(df_secondary.columns)}",
                f"Sample Reviews (First 10 rows):\n{df_secondary.head(10).to_string(index=False)}"
            ])
            if 'Rating' in df_secondary.columns:
                context_parts.append(f"Rating Distribution:\n{df_secondary['Rating'].value_counts().to_string()}")

        full_context = "\n\n".join(context_parts)

        prompt_instructions = f"""
You are an intelligent Business Intelligence Assistant chatting with an analyst. You have full access to the uploaded datasets:

{full_context}

USER QUERY: "{user_message}"

TASK & OUTPUT INSTRUCTIONS:
1. Answer the query thoroughly, concisely, and accurately based on the datasets. Use clean HTML tags (like <b>, <i>, <ul>, <li>, <p>) in the "reply" string.
2. If the user asks for a chart/graph/plot, OR if visualizing the metric breakdown/trend provides immediate value, construct a valid Chart.js data object.
3. Return STRICTLY a valid JSON object matching this schema (do NOT add any extra text outside the JSON):

{{
  "reply": "<HTML text response answering the user question>",
  "chart": {{
    "type": "bar" | "line" | "pie" | "doughnut",
    "title": "Chart Title Here",
    "labels": ["Label1", "Label2", ...],
    "datasets": [
      {{
        "label": "Dataset Name",
        "data": [10, 20, ...]
      }}
    ]
  }}
}}

If no chart is requested or needed, set "chart": null.
"""

        response = llm.invoke(prompt_instructions)
        raw_text = response.content[0].get('text', '') if isinstance(response.content, list) else response.content

        # Strip markdown syntax wrappers if returned
        clean_json_text = re.sub(r'```json\s*|\s*```', '', raw_text).strip()
        parsed_res = json.loads(clean_json_text)
        
        return jsonify(parsed_res)

    except Exception as e:
        # Graceful fallback response
        return jsonify({
            "reply": f"<p>I analyzed the dataset for your query: <b>{user_message}</b>.</p><p class='text-muted small'>Details: {str(e)}</p>",
            "chart": None
        })


if __name__ == "__main__":
    app.run(debug=True)