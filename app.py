"""
OVERLAP DETECTOR - Interactive Web App
======================================
Streamlit-based demo for Replit deployment
"""

import streamlit as st
import pandas as pd
import os
import sys
import tempfile
import random
import math
from pathlib import Path

# Import the core detector
from overlap_detector import OverlapDetector

# Page configuration
st.set_page_config(
    page_title="Overlap Detector",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main { background-color: #1a1a2e; }
    .stApp { background-color: #1a1a2e; }
    h1, h2, h3, p, div { color: white !important; }
    .metric-card {
        background: linear-gradient(135deg, #16213e 0%, #0f3460 100%);
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #e94560;
        margin: 10px 0;
    }
    .success-box {
        background: #00ff8820;
        border-left: 4px solid #00ff88;
        padding: 15px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'detector' not in st.session_state:
    st.session_state.detector = None
if 'results' not in st.session_state:
    st.session_state.results = None

# Header
st.title("🛣️ Overlap Detection System")
st.markdown("**GeoPandas (Precise) + DBSCAN (ML) Comparison**")

# Sidebar
with st.sidebar:
    st.header("📂 Data Source")
    
    data_source = st.radio(
        "Choose data source:",
        ["Upload WKT File", "Use Sample Data", "Generate Test Data"]
    )
    
    wkt_file_path = None
    
    if data_source == "Upload WKT File":
        uploaded_file = st.file_uploader(
            "Upload .wkt file",
            type=['wkt'],
            help="Upload a WKT file containing LINESTRING geometries"
        )
        
        if uploaded_file:
            # Save to temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.wkt', delete=False) as f:
                f.write(uploaded_file.getvalue().decode())
                wkt_file_path = f.name
            
            st.success(f"✓ Uploaded: {uploaded_file.name}")
            
            # Preview
            with st.expander("📄 Preview (first 10 lines)"):
                lines = uploaded_file.getvalue().decode().split('\n')[:10]
                st.code('\n'.join(lines))
    
    elif data_source == "Use Sample Data":
        samples = ["sample_data.wkt", "map_data.wkt"]
        sample_files = [f for f in samples if os.path.exists(f)]
        
        if sample_files:
            selected = st.selectbox("Select sample:", sample_files)
            wkt_file_path = selected
            st.success(f"✓ Selected: {selected}")
        else:
            st.warning("No sample files found. Please upload a file.")
    
    else:  # Generate Test Data
        st.subheader("🧪 Generate Synthetic Data")
        
        num_segments = st.slider("Number of segments:", 10, 200, 50)
        num_overlaps = st.slider("Number of overlaps:", 0, 50, 10)
        box_size = st.number_input("Coordinate box size:", 1000, 50000, 10000)
        
        if st.button("Generate Test Data", type="primary"):
            # Generate synthetic WKT
            def generate_test_wkt(n_segments, n_overlaps, box):
                lines = []
                
                # Generate regular segments
                for i in range(n_segments - n_overlaps):
                    x1, y1 = random.uniform(0, box), random.uniform(0, box)
                    x2, y2 = x1 + random.uniform(-box/10, box/10), y1 + random.uniform(-box/10, box/10)
                    lines.append(f"LINESTRING({x1} {y1}, {x2} {y2})")
                
                # Generate overlapping pairs
                for i in range(n_overlaps):
                    x1, y1 = random.uniform(0, box), random.uniform(0, box)
                    x2, y2 = x1 + random.uniform(-box/10, box/10), y1 + random.uniform(-box/10, box/10)
                    
                    # Original
                    lines.append(f"LINESTRING({x1} {y1}, {x2} {y2})")
                    
                    # Overlapping copy (slightly offset)
                    offset = random.uniform(-1, 1)
                    lines.append(f"LINESTRING({x1+offset} {y1+offset}, {x2+offset} {y2+offset})")
                
                return '\n'.join(lines)
            
            wkt_content = generate_test_wkt(num_segments, num_overlaps, box_size)
            
            # Save to temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.wkt', delete=False) as f:
                f.write(wkt_content)
                wkt_file_path = f.name
            
            st.success(f"✓ Generated {num_segments} segments with ~{num_overlaps*2} overlapping lines")
            st.session_state.generated_wkt = wkt_content
    
    st.divider()
    
    st.header("⚙️ Analysis Options")
    
    run_method = st.radio(
        "Detection method:",
        ["GeoPandas Only", "DBSCAN Only", "Both (Comparison)"]
    )
    
    st.divider()
    
    # Coordinate System Selection
    st.header("🌐 Coordinate System")
    
    coord_system = st.radio(
        "Select your data's coordinate format:",
        ["Projected (meters)", "Geographic (lat/lon)", "Auto-detect"],
        index=0,  # Default: Projected
        help="Choose the coordinate system matching your WKT data"
    )
    
    # Info display based on selection
    if coord_system == "Projected (meters)":
        st.info("""
        **📐 Projected Coordinates (Default)**
        - Values like: `LINESTRING(7071 8585, 7074 8588)`
        - Units: meters, feet, or local grid units
        - Buffer size: 1.0 = ~1 meter tolerance
        - Best for: UTM, State Plane, local survey grids
        """)
        default_buffer = 1.0
        buffer_max = 10.0
        buffer_step = 0.1
        
    elif coord_system == "Geographic (lat/lon)":
        st.info("""
        **🌍 Geographic Coordinates (WGS84)**
        - Values like: `LINESTRING(0.12 52.21, 0.13 52.22)`
        - Units: decimal degrees
        - Buffer size: 0.00005 ≈ ~5 meters at mid-latitudes
        - Best for: GPS data, OpenStreetMap, Google Maps exports
        """)
        default_buffer = 0.00005
        buffer_max = 0.001
        buffer_step = 0.00001
        
    else:  # Auto-detect
        st.warning("""
        **🔍 Auto-detect Mode**
        - System will analyze coordinate ranges
        - Coordinates < 180: likely Geographic (degrees)
        - Coordinates > 180: likely Projected (meters)
        - ⚠️ May not work for unusual local grids
        """)
        default_buffer = 1.0  # Will be overridden
        buffer_max = 10.0
        buffer_step = 0.1
    
    # Store selection in session state
    st.session_state.coord_system = coord_system
    
    st.divider()
    
    # Buffer Tolerance Settings
    st.header("🔧 Tolerance Settings")
    
    with st.expander("Configure Detection Thresholds", expanded=False):
        st.markdown("**Adjust thresholds for your coordinate system:**")
        
        st.subheader("Geometric Detection")
        buffer_size = st.slider(
            "Buffer Size (coordinate units)",
            min_value=buffer_step, max_value=buffer_max, value=default_buffer, step=buffer_step,
            help=f"Spatial buffer for near-overlaps. Default: {default_buffer}"
        )
        
        st.subheader("ML Detection (DBSCAN)")
        angle_tolerance = st.slider(
            "Angle Tolerance (degrees)",
            min_value=1, max_value=45, value=15, step=1,
            help="Max angle difference to consider as potential duplicate. Default: 15°"
        )
        
        distance_tolerance = st.slider(
            "Perpendicular Distance (units)",
            min_value=0.5, max_value=20.0, value=5.0, step=0.5,
            help="Max perpendicular offset between centroids. Default: 5.0"
        )
        
        overlap_threshold = st.slider(
            "Overlap Ratio Threshold",
            min_value=0.1, max_value=1.0, value=0.3, step=0.05,
            help="Min buffer intersection ratio to flag as overlap. Default: 0.3 (30%)"
        )
        
        st.subheader("DBSCAN Clustering")
        dbscan_eps = st.slider(
            "DBSCAN eps (feature space)",
            min_value=0.1, max_value=2.0, value=0.5, step=0.1,
            help="Clustering distance threshold. Default: 0.5"
        )
        
        dbscan_min_samples = st.slider(
            "DBSCAN min_samples",
            min_value=1, max_value=10, value=2, step=1,
            help="Minimum points to form a cluster. Default: 2"
        )
        
        st.markdown("---")
        st.info("💡 **Tip**: Lower thresholds = more detections (higher recall, more false positives)")
    
    # Store settings in session state
    st.session_state.tolerance_settings = {
        'buffer_size': buffer_size if 'buffer_size' in dir() else 1.0,
        'angle_tolerance': angle_tolerance if 'angle_tolerance' in dir() else 15,
        'distance_tolerance': distance_tolerance if 'distance_tolerance' in dir() else 5.0,
        'overlap_threshold': overlap_threshold if 'overlap_threshold' in dir() else 0.3,
        'dbscan_eps': dbscan_eps if 'dbscan_eps' in dir() else 0.5,
        'dbscan_min_samples': dbscan_min_samples if 'dbscan_min_samples' in dir() else 2
    }

# Main area
if wkt_file_path:
    tabs = st.tabs(["📊 Analysis", "🗺️ Map", "📈 Comparison", "📝 Feedback", "💾 Export"])
    
    with tabs[0]:  # Analysis
        st.header("Analysis Dashboard")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if st.button("🚀 Run Detection", type="primary", use_container_width=True):
                with st.spinner("Loading data..."):
                    # Map UI selection to code parameter
                    coord_map = {
                        "Projected (meters)": "projected",
                        "Geographic (lat/lon)": "geographic",
                        "Auto-detect": "auto"
                    }
                    coord_param = coord_map.get(st.session_state.get('coord_system', 'Projected (meters)'), 'projected')
                    
                    # For auto-detect, pass None so detector uses auto-detected value
                    # For manual selection, use the slider value
                    actual_buffer = None if coord_param == "auto" else buffer_size
                    
                    detector = OverlapDetector(
                        wkt_file_path, 
                        buffer_size=actual_buffer,
                        coord_system=coord_param
                    )
                    st.session_state.detector = detector
                    
                    # Show detected parameters
                    st.info(f"📍 Using buffer size: **{detector.buffer_size}**")
                
                if run_method in ["GeoPandas Only", "Both (Comparison)"]:
                    with st.spinner("Running GeoPandas detection..."):
                        detector.run_geometric_detection()
                
                if run_method in ["DBSCAN Only", "Both (Comparison)"]:
                    with st.spinner("Running DBSCAN detection..."):
                        detector.run_ml_detection()
                
                # Generate all outputs (CSV, map, report)
                with st.spinner("Generating outputs..."):
                    detector.generate_outputs()
                
                st.session_state.results = {
                    'geo_count': getattr(detector, 'geo_count', len(detector.overlaps)),
                    'geo_time': getattr(detector, 'geo_time', 0),
                    'ml_count': getattr(detector, 'ml_count', 0),
                    'ml_time': getattr(detector, 'ml_time', 0),
                    'overlaps': detector.overlaps
                }
                
                # Store map HTML for display
                map_file = f"{detector.basename}_map.html"
                if os.path.exists(map_file):
                    with open(map_file, 'r') as f:
                        st.session_state.map_html = f.read()
                
                st.success("✅ Analysis Complete! Check the Map tab.")
        
        with col2:
            if st.session_state.detector:
                segments = len(st.session_state.detector.gdf)
                st.metric("Total Segments", segments)
        
        # Results
        if st.session_state.results:
            results = st.session_state.results
            
            st.subheader("🎯 Detection Results")
            
            if run_method in ["GeoPandas Only", "Both (Comparison)"]:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🔷 GeoPandas (Geometric)</h3>
                    <p style="font-size: 32px; color: #00ff88;">{results['geo_count']}</p>
                    <p>Overlaps Detected</p>
                    <p style="font-size: 14px; color: #aaa;">Time: {results['geo_time']:.3f}s</p>
                </div>
                """, unsafe_allow_html=True)
            
            if run_method in ["DBSCAN Only", "Both (Comparison)"]:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🟣 DBSCAN (ML)</h3>
                    <p style="font-size: 32px; color: #9b59b6;">{results['ml_count']}</p>
                    <p>Overlaps Detected</p>
                    <p style="font-size: 14px; color: #aaa;">Time: {results['ml_time']:.3f}s</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Overlaps table
            if results['overlaps']:
                st.subheader("📋 Detected Overlaps")
                df = pd.DataFrame(results['overlaps'])
                display_cols = [c for c in ['id1', 'id2', 'type', 'confidence', 'method'] if c in df.columns]
                st.dataframe(df[display_cols], use_container_width=True)
    
    with tabs[1]:  # Map
        st.header("Interactive Map")
        
        if 'map_html' in st.session_state and st.session_state.map_html:
            # Display the map immediately
            st.components.v1.html(st.session_state.map_html, height=700, scrolling=True)
            
            st.markdown("""
            **Legend:**
            - 🔘 Gray lines = All segments  
            - 🟣 Purple lines = Overlapping pairs  
            - 🔴 Red = High confidence (≥90%)  
            - 🟠 Orange = Medium (70-90%)  
            - 🟡 Yellow = Low (<70%)
            """)
        elif st.session_state.detector:
            st.info("⬆️ Click 'Run Detection' first to generate the map")
        else:
            st.info("⬆️ Select a data source and run analysis to see the map")
    
    with tabs[2]:  # Comparison
        st.header("Method Comparison")
        
        if run_method == "Both (Comparison)" and st.session_state.detector:
            detector = st.session_state.detector
            
            if st.button("📊 Generate Comparison Report"):
                with st.spinner("Generating comparison..."):
                    detector.generate_comparison_report()
                    
                    comp_file = f"{detector.basename}_comparison.html"
                    
                    if os.path.exists(comp_file):
                        with open(comp_file, 'r') as f:
                            comp_html = f.read()
                        
                        st.components.v1.html(comp_html, height=800, scrolling=True)
        else:
            st.info("💡 Select 'Both (Comparison)' mode to see comparison")
    
    with tabs[3]:  # Feedback
        st.header("📝 Feedback Training")
        
        st.markdown("""
        ### 🔄 Fine-Tuning Procedure
        
        The system learns from your corrections to improve future detections.
        
        ---
        
        **Step 1: Download Feedback Template**
        - Click the download button below to get a CSV with detected overlaps
        - The file contains columns: `id1`, `id2`, `type`, `confidence`, `is_overlap`, `comments`
        
        **Step 2: Review Each Detection**
        - Open the CSV in Excel or Google Sheets
        - For each row, fill the `is_overlap` column:
          - `TRUE` → This IS a real overlap (correct detection)
          - `FALSE` → This is NOT an overlap (false positive)
        - Optionally add notes in the `comments` column
        
        **Step 3: Upload Corrected File**
        - Save your CSV and upload it below
        - Click "Train from Feedback" to improve the model
        
        ---
        
        **📊 How Training Works:**
        
        | Your Label | System Action |
        |------------|---------------|
        | `TRUE` | Reinforces detection patterns |
        | `FALSE` | Adjusts thresholds to avoid similar false positives |
        
        The system analyzes the geometric features (angle, distance, overlap ratio) 
        of your corrections and learns optimal decision boundaries.
        """)
        
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📥 Step 1: Download Template")
            if st.session_state.detector:
                detector = st.session_state.detector
                feedback_file = f"{detector.basename}_feedback.csv"
                
                if os.path.exists(feedback_file):
                    st.download_button(
                        "⬇️ Download Feedback CSV",
                        open(feedback_file, 'r').read(),
                        file_name=feedback_file,
                        mime='text/csv',
                        use_container_width=True
                    )
                    st.caption("Contains: id1, id2, type, confidence, is_overlap, comments")
                else:
                    st.warning("Run detection first to generate the feedback template")
            else:
                st.info("Run detection first to generate feedback template")
        
        with col2:
            st.subheader("📤 Step 3: Upload Corrected File")
            uploaded_feedback = st.file_uploader("Upload Corrected Feedback CSV", type=['csv'])
            
            if uploaded_feedback:
                # Preview uploaded file
                feedback_df = pd.read_csv(uploaded_feedback)
                st.caption(f"Uploaded: {len(feedback_df)} rows")
                
                # Check for is_overlap column filled
                filled = feedback_df['is_overlap'].notna() & (feedback_df['is_overlap'] != '')
                st.caption(f"Labeled: {filled.sum()} / {len(feedback_df)}")
                
                # Reset file pointer for saving
                uploaded_feedback.seek(0)
                
                with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                    f.write(uploaded_feedback.getvalue().decode())
                    feedback_path = f.name
                
                if st.button("🎓 Train from Feedback", type="primary", use_container_width=True):
                    with st.spinner("Training model from your feedback..."):
                        detector.train_from_feedback(feedback_path)
                    st.success("✅ Training complete! Model updated with your corrections.")
    
    with tabs[4]:  # Export
        st.header("Export Results")
        
        if st.session_state.detector:
            detector = st.session_state.detector
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                csv_file = f"{detector.basename}_overlaps.csv"
                if os.path.exists(csv_file):
                    st.download_button(
                        "📊 Download CSV",
                        open(csv_file, 'r').read(),
                        file_name=csv_file,
                        mime='text/csv',
                        use_container_width=True
                    )
            
            with col2:
                report_file = f"{detector.basename}_report.html"
                if os.path.exists(report_file):
                    st.download_button(
                        "📄 Download Report",
                        open(report_file, 'r').read(),
                        file_name=report_file,
                        mime='text/html',
                        use_container_width=True
                    )
            
            with col3:
                map_file = f"{detector.basename}_map.html"
                if os.path.exists(map_file):
                    st.download_button(
                        "🗺️ Download Map",
                        open(map_file, 'r').read(),
                        file_name=map_file,
                        mime='text/html',
                        use_container_width=True
                    )

else:
    st.info("👈 Please select a data source from the sidebar to begin")
    
    # Instructions
    st.markdown("""
    ## Getting Started
    
    1. **Choose Data Source**
       - Upload your own WKT file
       - Use pre-loaded samples
       - Generate synthetic test data
    
    2. **Run Analysis**
       - Select detection method (GeoPandas, DBSCAN, or both)
       - Click "Run Detection"
    
    3. **View Results**
       - Interactive map with color-coded overlaps
       - Method comparison (if both selected)
       - Export CSV/HTML reports
    
    4. **Train from Feedback**
       - Download feedback template
       - Mark true/false positives
       - Upload to improve accuracy
    """)
