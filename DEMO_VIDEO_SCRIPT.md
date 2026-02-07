# 🎥 Map Overlap Detection - Demo Video Script

**Target Audience:** Hackathon Judges / Technical Evaluators  
**Goal:** Showcase how the tool solves the "Needle in a Haystack" problem of map errors using Hybrid AI.  
**Duration:** ~2-3 Minutes

---

## 🎬 1. THE HOOK: "Why Does This Matter?" (0:00 - 0:30)

**Visual:** 
- Show a messy map with overlapping lines (use `synthetic_100.wkt` or `synthetic_duplicates.wkt` visualization).
- Overlay text: *"Finding duplicates manually is impossible."*

**Narration:**
> "Imagine finding a duplicate road in a city map with 10,000 streets. It's like finding a needle in a haystack. 
> Manual QA is slow, expensive, and error-prone. One missed duplicate can break navigation routing for thousands of users.
> Today, **Team ByteForce** presents an intelligent, automated solution to detect map overlaps instantly."

---

## 💡 2. THE CHALLENGE & INNOVATION (0:30 - 1:00)

**Visual:** 
- Slide showing "Exact vs. Fuzzy" (use a diagram or the `problem2_qa_report.md` concept).
- Split screen: GeoPandas logo (Math) | DBSCAN logo (AI).

**Narration:**
> "The core challenge isn't just finding identical lines—that's easy. 
> The real problem is **'Fuzzy Duplicates'**: lines that are slightly offset due to GPS noise or different data sources.
> Traditional tools miss these.
> 
> **How Our ML Engine Works:**
> "But how do we find fuzzy duplicates? We don't just look at lines.
> We transform every pair of lines into **4 Feature Parameters**:
> 1. **Angle Difference:** Do they point the same way?
> 2. **Perpendicular Distance:** How far apart are they?
> 3. **Length Ratio:** Are they similar size?
> 4. **Proximity Score:** Do they cover the same area?
> 
> We feed these 4 parameters into **DBSCAN Clustering**. 
> Instead of writing hard rules like 'if dist < 5m', DBSCAN automatically finds the 'cluster' of high-similarity segments.
> This allows us to catch duplicates that are 5 meters apart, slightly rotated, or fragmented—things that strict geometry checks miss."

---

## 🚀 3. LIVE DEMO: "From Chaos to Clarity" (1:00 - 2:00)

**Visual:** Screen recording of the **Streamlit App**.

**Step 1: Upload Data**
> "Let's see it in action. We drag and drop a raw WKT file. Notice our **Auto-Detection** feature instantly identifies the coordinate system—handling both Meters and Lat/Lon automatically."
*Action:* Show upload of `Large_map.wkt`. Point to "📍 Auto-detected: Geographic" message.

**Step 2: Analysis Running**
> "In seconds, the system runs both engines parallelly."
*Action:* Click 'Run Detection'. Show progress bars.

**Step 3: The Interactive Map**
> "Here is the result. The interactive map highlights errors in **RED**.
> See this? It looks like one road, but zooming in reveal it's actually two overlapping segments. The ML engine caught this subtle offset that standard tools missed."
*Action:* Zoom into a cluster of red lines. Click a segment to show the popup details.

**Step 4: Comparison Report**
> "We provide a transparent Comparison Report. You can see exactly what GeoPandas found versus what ML found, building trust in the AI's results."
*Action:* Switch to 'Comparison Report' tab. Show the Venn diagram/table.

---

## 🔄 4. THE FEEDBACK LOOP (2:00 - 2:30)

**Visual:** Show the 'Feedback' tab / CSV editing.

**Narration:**
> "But AI isn't perfect. That's why we built a **Human-in-the-Loop** workflow.
> A QA expert can download the report, mark 'False Positives', and upload it back.
> The system **learns** from this feedback, tuning its thresholds for the next run. It gets smarter with every use."

---

## 🏆 5. CONCLUSION (2:30 - 2:45)

**Visual:** Summary Slide (Speed, Accuracy, Actionable).

**Narration:**
> "Team ByteForce has built a solution that is:
> 1. **Fast:** Processes thousands of segments in seconds.
> 2. **Intelligent:** Auto-detects coordinate systems and fuzzy duplicates.
> 3. **Actionable:** Exports ready-to-use reports for map editors.
> 
> We are democratizing GIS Quality Assurance."

---

## 🛠️ PRODUCING THE VIDEO

**Suggested Recording Flow:**
1. **Record Screen:** Run the app locally. Record the full workflow (Upload -> Analyze -> Map -> Report).
2. **Key Moments to Zoom In:**
   - The "Auto-detected" message.
   - The Map visualization (especially zooming in to show parallel lines).
   - The "DBSCAN Overlaps" count in the sidebar.
3. **Voiceover:** Read the script naturally, pausing for the visual actions.
