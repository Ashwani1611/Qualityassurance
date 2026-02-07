# Overlap Detection Prototype 🛣️

A dual-method system for detecting street geometry overlaps using **GeoPandas** (computational geometry) and **DBSCAN** (machine learning).

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run standard detection
python overlap_detector.py sample_data.wkt

# Run with method comparison (GeoPandas vs DBSCAN)
python overlap_detector.py sample_data.wkt --compare
```

---

## Adding Your Test Data

**Option 1: Place in this folder**
```bash
cp /path/to/your/streets.wkt ./
python overlap_detector.py streets.wkt
```

**Option 2: Use full path**
```bash
python overlap_detector.py /full/path/to/your/streets.wkt
```

**WKT Format Expected:**
```
LINESTRING(x1 y1, x2 y2, x3 y3, ...)
LINESTRING(x1 y1, x2 y2)
...
```

---

## Method Comparison

| Metric | GeoPandas | DBSCAN |
|--------|-----------|--------|
| **Type** | Exact Geometry | Statistical Clustering |
| **Best For** | QA Validation | Fuzzy Duplicates |
| **Precision** | 100% (Mathematical) | ~70-90% (Learned) |
| **Speed** | Faster | Slower |

Run `--compare` flag to generate comparison report.

---

## Feedback Training (Supervised Learning)

### How It Works:
1. **Run detection** → Generates `*_feedback.csv`
2. **Open CSV** → Mark `is_overlap` column as `TRUE` or `FALSE`
3. **Run training** → System learns from your corrections

### Commands:
```bash
# Step 1: Generate feedback template
python overlap_detector.py streets.wkt

# Step 2: Edit streets_feedback.csv (mark TRUE/FALSE)

# Step 3: Train from corrections
python overlap_detector.py streets.wkt --train streets_feedback.csv
```

### Future Improvements:
With labeled data, the system can train a **Random Forest** or **XGBoost** classifier for higher accuracy predictions.

---

## Output Files

| File | Description |
|------|-------------|
| `*_map.html` | Interactive map (dark theme, color-coded) |
| `*_overlaps.csv` | Data export for spreadsheet analysis |
| `*_report.html` | Summary report |
| `*_feedback.csv` | QA feedback template |
| `*_comparison.html` | Method comparison (with `--compare`) |

---

## Color Coding

| Color | Meaning |
|-------|---------|
| **Gray** | All segments |
| **Purple** | Segments involved in overlaps |
| **Red** | High confidence overlap (≥90%) |
| **Orange** | Medium confidence (70-90%) |
| **Yellow** | Low confidence (<70%) |
