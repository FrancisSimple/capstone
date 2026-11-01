"""
Step 2 -- documented literature search (template Section 3.1).

Runs every search string against the OpenAlex index (title + abstract field) and
records the date, exact query, filters, number of records, and the top records by
relevance. Screening decisions are recorded separately in
paper/analysis/outputs/lit_screening.csv (filled in by the reviewers).

    paper/.venv/Scripts/python paper/analysis/02_literature_search.py
"""
import csv
import datetime as dt
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "analysis" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
API = "https://api.openalex.org/works"
FILTERS = "type:article|review|preprint|book-chapter,publication_year:2000-2026"
TOP_N = 25

# Q1-Q6 are the template's search combinations; Q7-Q14 extend them to the gaps
# the Step 1 audit exposed (low-cost gas sensors, leakage, label scarcity, Ghana).
QUERIES = {
    "Q1": '"postharvest" AND "uncertainty quantification" AND "shelf life"',
    "Q2": '("orange spoilage" AND "survival analysis") OR ("orange spoilage" AND "time-to-event")',
    "Q3": '"sensor missingness" AND "food quality prediction"',
    "Q4": '"climate data" AND "postharvest loss" AND "sensor fusion"',
    "Q5": '"probabilistic calibration" AND "food spoilage prediction"',
    "Q6": '"citrus" AND "digital twin" AND "remaining shelf life"',
    # Relaxed forms of Q1-Q6: the exact template phrases return almost nothing, which
    # would overstate novelty; these keep the concepts but drop the rigid phrasing.
    "Q1r": '("postharvest" OR "post-harvest") AND ("uncertainty" OR "stochastic") AND "shelf life" AND ("prediction" OR "model")',
    "Q2r": '("orange" OR "citrus" OR "fruit") AND ("survival analysis" OR "time-to-event" OR "hazard model") AND ("decay" OR "spoilage" OR "shelf life")',
    "Q3r": '("missing data" OR "missing values" OR "sensor failure") AND ("food quality" OR "shelf life" OR "food spoilage")',
    "Q4r": '("weather" OR "climate") AND ("postharvest loss" OR "post-harvest loss") AND ("model" OR "sensor")',
    "Q5r": '("calibration" OR "probabilistic") AND ("food spoilage" OR "shelf life") AND ("machine learning" OR "prediction")',
    "Q6r": '("orange" OR "citrus") AND ("digital twin" OR "digital replica") AND ("shelf life" OR "postharvest")',
    "Q7": '"shelf life" AND ("uncertainty" OR "probabilistic") AND ("fruit" OR "citrus")',
    "Q8": '("orange" OR "citrus") AND ("electronic nose" OR "gas sensor") AND ("decay" OR "spoilage" OR "quality")',
    "Q9": '("metal oxide" OR "MQ-135" OR "MQ135" OR "low-cost gas sensor") AND ("fruit" OR "food spoilage")',
    "Q10": '("data leakage" OR "pseudoreplication" OR "grouped cross-validation") AND ("food" OR "agriculture")',
    "Q11": '("postharvest loss" OR "post-harvest loss") AND ("Ghana" OR "sub-Saharan Africa") AND ("fruit" OR "citrus" OR "orange")',
    "Q12": '("cold chain" OR "storage") AND ("temperature" ) AND ("shelf life prediction") AND ("fruit")',
    "Q13": '("missing data" OR "sensor failure" OR "sensor fault") AND ("cold chain" OR "food monitoring")',
    "Q14": '("semi-supervised" OR "self-training" OR "label scarcity") AND ("shelf life" OR "food freshness")',
}


def get(params):
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "capstone-lit-search (academic research)"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def abstract(inv):
    if not inv:
        return ""
    pos = [(p, w) for w, ps in inv.items() for p in ps]
    return " ".join(w for _, w in sorted(pos))


log, records = [], []
today = dt.date.today().isoformat()
for qid, q in QUERIES.items():
    params = {"filter": f"title_and_abstract.search:{q},{FILTERS}", "per-page": TOP_N,
              "sort": "relevance_score:desc",
              "select": "id,doi,title,publication_year,primary_location,cited_by_count,abstract_inverted_index,authorships,type"}
    try:
        res = get(params)
        n = res["meta"]["count"]
    except Exception as e:  # keep the log honest about failures
        log.append({"id": qid, "query": q, "date": today, "database": "OpenAlex", "records": f"ERROR {e}"})
        continue
    log.append({"id": qid, "query": q, "date": today, "database": "OpenAlex (title+abstract)",
                "filters": FILTERS, "records": n})
    print(f"{qid}: {n:6d}  {q}")
    for rank, w in enumerate(res["results"], 1):
        src = (w.get("primary_location") or {}).get("source") or {}
        authors = [a["author"]["display_name"] for a in w.get("authorships", [])]
        records.append({"query": qid, "rank": rank, "openalex": w["id"], "doi": w.get("doi") or "",
                        "year": w.get("publication_year"), "title": w.get("title") or "",
                        "venue": src.get("display_name") or "", "type": w.get("type"),
                        "cited_by": w.get("cited_by_count"),
                        "authors": "; ".join(authors[:6]) + (" et al." if len(authors) > 6 else ""),
                        "abstract": abstract(w.get("abstract_inverted_index"))})
    time.sleep(0.3)

with open(OUT / "lit_search_log.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["id", "database", "date", "query", "filters", "records"])
    w.writeheader()
    w.writerows(log)

# deduplicate on OpenAlex id, keeping the first query that retrieved each work
seen, dedup = set(), []
for r in records:
    if r["openalex"] not in seen:
        seen.add(r["openalex"])
        dedup.append(r)
with open(OUT / "lit_candidates.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
    w.writeheader()
    w.writerows(dedup)
print(f"retrieved {len(records)} top records, {len(dedup)} after de-duplication")
