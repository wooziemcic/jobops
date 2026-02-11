# app/pages/02_Job_Feed.py
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import time
from datetime import datetime, timezone
from collections import defaultdict

import streamlit as st

from app.storage.jobs_store import write_raw, write_latest_normalized, read_latest_normalized
from app.connectors.rss import fetch_rss_feed, normalize_rss_entries
from app.connectors.jsearch import fetch_jsearch_jobs
from app.connectors.jsearch_normalize import normalize_jsearch_payload
from app.core.jobs_utils import (
    dedupe_jobs,
    is_early_career,
    is_us_job,
    extract_us_state_code,
    ROLE_BUCKETS,
)

st.set_page_config(page_title="Job Feed", layout="wide")
st.title("Job Feed")
st.caption("JSearch (ATS/aggregator via RapidAPI) + Curated RSS. Fresh, deduped, apply links first.")
st.divider()

# ---------------------------
# Curated RSS Feeds (editable)
# ---------------------------
DEFAULT_FEEDS = [
    ["Amazon (Data/Analytics)", "https://www.amazon.jobs/en/search.rss?category[]=data-science&category[]=analytics"],
    ["Goldman Sachs", "https://www.goldmansachs.com/careers/search/rss"],
    ["JPMorgan Chase", "https://careers.jpmorgan.com/us/en/jobs/rss"],
    ["Capital One", "https://www.capitalonecareers.com/search-jobs/rss"],
    ["Deloitte", "https://apply.deloitte.com/careers/SearchJobs.rss"],
    ["PwC", "https://jobs.us.pwc.com/search.rss"],
    ["EY", "https://careers.ey.com/ey/search/rss"],
    ["KPMG", "https://www.kpmgcareers.com/rss"],
    ["Pfizer (Workday)", "https://pfizer.wd1.myworkdayjobs.com/PfizerCareers/rss"],
    ["Johnson & Johnson", "https://jobs.jnj.com/jobs/rss"],
    ["Novartis", "https://jobs.novartis.com/jobs/rss"],
    ["Moderna (Workday)", "https://modernatx.wd1.myworkdayjobs.com/careers/rss"],
    ["GSK", "https://jobs.gsk.com/jobs/rss"],
]

with st.expander("RSS Feeds (curated)", expanded=True):
    st.write("Edit the curated RSS list as JSON: [[name, url], ...]")
    feeds_text = st.text_area(
        "Feeds",
        value=json.dumps(DEFAULT_FEEDS, indent=2),
        height=260,
    )

# ---------------------------
# JSearch Config (RapidAPI)
# ---------------------------
with st.expander("JSearch (RapidAPI)", expanded=True):
    st.write("Credentials should be set in Streamlit secrets (recommended).")

    secret_key = st.secrets.get("RAPIDAPI_KEY", "")
    secret_host = st.secrets.get("RAPIDAPI_HOST", "jsearch.p.rapidapi.com")

    rapidapi_key = st.text_input("RAPIDAPI_KEY", value=secret_key, type="password")
    rapidapi_host = st.text_input("RAPIDAPI_HOST", value=secret_host, type="default")

    country = st.text_input("Country code", value="us")

    col1, col2 = st.columns(2)
    with col1:
        start_page = st.selectbox("Start page", options=[1, 2, 3, 5, 10], index=0)
    with col2:
        num_pages = st.selectbox("Pages to fetch", options=[1, 2, 3, 5], index=1)

# ---------------------------
# Filters
# ---------------------------
st.divider()

col_a, col_b, col_c = st.columns([1, 1, 2], gap="large")

with col_a:
    bucket_filter = st.multiselect(
        "Role buckets",
        options=ROLE_BUCKETS,
        default=["DA", "DS", "BA", "Quant", "ML", "AI"],
    )

with col_b:
    keyword_filter = st.text_input("Keyword filter (optional)", value="")

with col_c:
    st.caption("Tip: Keep RSS curated. JSearch handles breadth; RSS adds high-signal targets.")
    early_only = st.checkbox("Early-career only (0–4 yrs)", value=True)
    us_only = st.checkbox("United States only", value=True)

# ---------------------------
# Refresh action
# ---------------------------
st.divider()
btn = st.button("Refresh jobs now", type="primary")

if btn:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    all_jobs = []

    # RSS ingestion
    feeds = json.loads(feeds_text) if feeds_text.strip() else []
    for name, url in feeds:
        try:
            rss_result = fetch_rss_feed(url)

            # Supports both styles:
            # - dict (old)
            # - (parsed, fetch_meta) (new)
            if isinstance(rss_result, tuple) and len(rss_result) == 2:
                parsed, fetch_meta = rss_result
                raw_payload = {"fetch_meta": fetch_meta, "parsed": parsed}
            else:
                parsed = rss_result
                raw_payload = parsed

            write_raw(f"rss_{name}", raw_payload, ts)

            jobs = normalize_rss_entries(name, parsed)
            all_jobs.extend(jobs)

            if parsed.get("bozo") and len(parsed.get("entries", [])) == 0:
                st.warning(
                    f"RSS feed '{name}' malformed and returned 0 entries: {parsed.get('bozo_exception','')}"
                )
        except Exception as e:
            st.error(f"RSS ingestion error for '{name}': {e}")

    # JSearch ingestion (bucketed, retries)
    if not rapidapi_key:
        st.warning("RapidAPI key missing. Set RAPIDAPI_KEY in Streamlit secrets (recommended).")
    else:
        bucket_to_query = {
            "DA": "data analyst",
            "BA": "business analyst",
            "DS": "data scientist",
            "ML": "machine learning engineer",
            "AI": "ai engineer",
            "Quant": "quantitative analyst",
            "Other": "analyst",
        }

        early_tokens = "(entry OR junior OR associate OR analyst OR new grad OR early career OR 0-3 OR 1-3 OR 2-4)"

        selected = bucket_filter or ["DA", "BA", "DS", "ML", "AI", "Quant"]
        selected = [b for b in selected if b in bucket_to_query]

        for b in selected:
            q = f"({bucket_to_query[b]}) AND {early_tokens}"
            if keyword_filter.strip():
                q = f"{q} AND ({keyword_filter.strip()})"

            last_err = None
            for attempt in range(1, 4):
                try:
                    payload = fetch_jsearch_jobs(
                        query=q,
                        rapidapi_key=rapidapi_key.strip(),
                        host=rapidapi_host.strip(),
                        country=country.strip().lower(),
                        page=int(start_page),
                        num_pages=int(num_pages),
                        timeout=60,
                    )

                    write_raw(f"jsearch_{b}", payload, ts)

                    jobs = normalize_jsearch_payload(payload)
                    for j in jobs:
                        if not j.get("role_bucket"):
                            j["role_bucket"] = b

                    all_jobs.extend(jobs)
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    time.sleep(0.75 * attempt)

            if last_err is not None:
                st.warning(f"JSearch failed for bucket {b}: {last_err}")

    # Dedupe + Save-time filters
    all_jobs = dedupe_jobs(all_jobs)

    if early_only:
        all_jobs = [j for j in all_jobs if is_early_career(j)]
    if us_only:
        all_jobs = [j for j in all_jobs if is_us_job(j)]

    write_latest_normalized(all_jobs)
    st.success(f"Saved jobs_latest.json with {len(all_jobs)} jobs.")

# ---------------------------
# Load + Display
# ---------------------------
jobs = read_latest_normalized()

# Display-time safety filters
if early_only:
    jobs = [j for j in jobs if is_early_career(j)]
if us_only:
    jobs = [j for j in jobs if is_us_job(j)]
if bucket_filter:
    jobs = [j for j in jobs if (j.get("role_bucket") in bucket_filter)]
if keyword_filter.strip():
    qk = keyword_filter.strip().lower()

    def hit(j):
        return (
            qk in (j.get("title") or "").lower()
            or qk in (j.get("company") or "").lower()
            or qk in (j.get("description") or "").lower()
        )

    jobs = [j for j in jobs if hit(j)]

st.subheader(f"Jobs ({len(jobs)})")

# Group by company
grouped = defaultdict(list)
for j in jobs:
    grouped[j.get("company") or "Unknown"].append(j)

companies_sorted = sorted(grouped.items(), key=lambda kv: len(kv[1]), reverse=True)

# ---------- UI Upgrade A: clickable Apply links ----------
for company, items in companies_sorted:
    with st.expander(f"{company} ({len(items)})", expanded=False):
        rows = []
        for j in items:
            rows.append(
                {
                    "Bucket": j.get("role_bucket"),
                    "Title": j.get("title"),
                    "Location": j.get("location") or "",
                    "Source": j.get("source"),
                    "Apply": j.get("apply_url") or "",
                }
            )

        # Prefer LinkColumn if available; fallback to markdown list
        try:
            st.data_editor(
                rows,
                use_container_width=True,
                hide_index=True,
                disabled=True,
                column_config={
                    "Apply": st.column_config.LinkColumn(
                        "Apply",
                        help="Open the application link",
                        validate="^https?://.*",
                    )
                },
            )
        except Exception:
            for r in rows:
                apply_url = (r.get("Apply") or "").strip()
                title = (r.get("Title") or "").strip()
                loc = (r.get("Location") or "").strip()
                bucket = (r.get("Bucket") or "").strip()
                if apply_url.startswith("http"):
                    st.markdown(f"- **{title}** ({bucket}) · {loc} · [Apply]({apply_url})")
                else:
                    st.markdown(f"- **{title}** ({bucket}) · {loc}")

# ---------- UI Upgrade B: interactive US map (state counts) ----------
st.divider()
st.subheader("US job density map (by state)")

state_counts = defaultdict(int)
for j in jobs:
    code = extract_us_state_code(j.get("location") or "")
    if code:
        state_counts[code] += 1

if not state_counts:
    st.info("No state-coded locations found yet. RSS often lacks structured location; JSearch usually provides it.")
else:
    try:
        import pandas as pd
        import plotly.express as px

        df = pd.DataFrame(
            [{"state": k, "count": v} for k, v in sorted(state_counts.items(), key=lambda kv: kv[1], reverse=True)]
        )

        fig = px.choropleth(
            df,
            locations="state",
            locationmode="USA-states",
            color="count",
            scope="usa",
            hover_name="state",
            hover_data={"count": True},
        )
        st.plotly_chart(fig, use_container_width=True)

        # Helpful companion view
        st.caption("Top states (count)")
        st.dataframe(df.head(15), use_container_width=True, hide_index=True)
    except Exception:
        # Fallback if plotly/pandas missing
        st.bar_chart(dict(state_counts))

st.caption("Apply is the company/ATS link. jobs_latest.json is the input for Ranking next.")
