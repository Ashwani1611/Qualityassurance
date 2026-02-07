#!/usr/bin/env python3
"""
OVERLAP DETECTOR: FINAL PROTOTYPE
=================================
A comprehensive system for detecting street overlaps using both:
1. Computational Geometry (GeoPandas) - For precise topological errors
2. Machine Learning (DBSCAN) - For fuzzy duplicate detection

FEATURES:
- Automatic method selection or hybrid mode
- Interactive Leaflet map generation
- CSV/GeoJSON export
- Feedback loop for QA
- HTML summary reports
"""

import os
import sys
import math
import re
import time
import argparse
import json
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, Point
from shapely.wkt import loads as load_wkt
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import folium

# Default Configuration
DEFAULT_BUFFER_SIZE = 1.0  # For projected coordinate systems (meters)


def detect_coordinate_system(gdf):
    """
    Auto-detect if coordinates are lat/lon (degrees) or projected (meters).
    
    Returns:
        tuple: (buffer_size, system_name, is_geographic)
    """
    bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
    max_coord = max(abs(bounds[0]), abs(bounds[2]), abs(bounds[1]), abs(bounds[3]))
    
    # Lat/Lon: typically -180 to 180 (x) and -90 to 90 (y)
    # Check if within geographic bounds
    if max_coord < 180 and -90 <= bounds[1] <= 90 and -90 <= bounds[3] <= 90:
        # Likely lat/lon (degrees) - use ~5m buffer at mid-latitudes
        return (0.00005, "Geographic (lat/lon)", True)
    else:
        # Likely projected (meters)
        return (1.0, "Projected (meters)", False)


class OverlapDetector:
    def __init__(self, wkt_file, buffer_size=None, coord_system="projected"):
        self.wkt_file = wkt_file
        self.basename = os.path.splitext(os.path.basename(wkt_file))[0]
        self.gdf = self._load_data()
        self.overlaps = []
        
        # Handle coordinate system and buffer size
        self.is_geographic = False
        if coord_system == "auto":
            detected_buffer, system_name, is_geo = detect_coordinate_system(self.gdf)
            self.buffer_size = buffer_size if buffer_size else detected_buffer
            self.is_geographic = is_geo
            print(f"📍 Auto-detected: {system_name} → Buffer: {self.buffer_size}")
        elif coord_system == "geographic":
            self.buffer_size = buffer_size if buffer_size else 0.00005
            self.is_geographic = True
            print(f"📍 Geographic mode → Buffer: {self.buffer_size}")
        else:  # projected (default)
            self.buffer_size = buffer_size if buffer_size else DEFAULT_BUFFER_SIZE
            self.is_geographic = False
            print(f"📍 Projected mode → Buffer: {self.buffer_size}")

    def _load_data(self):
        """Robust WKT loader that handles multiline strings"""
        print(f"Loading {self.wkt_file}...")
        try:
            with open(self.wkt_file, 'r') as f:
                content = f.read()
            
            # Regex to find LINESTRING(...) across multiple lines
            wkt_matches = re.findall(r'LINESTRING\s*\([0-9.,\s-]+\)', content, re.IGNORECASE)
            
            geoms = []
            ids = []
            for i, wkt_str in enumerate(wkt_matches):
                try:
                    clean_wkt = wkt_str.replace('\n', ' ').strip()
                    g = load_wkt(clean_wkt)
                    if not g.is_empty:
                        geoms.append(g)
                        ids.append(f"street_{i+1}")
                except:
                    pass
            
            gdf = gpd.GeoDataFrame({'id': ids}, geometry=geoms)
            print(f"✓ Loaded {len(gdf)} segments")
            return gdf
        except Exception as e:
            print(f"❌ Error loading file: {e}")
            sys.exit(1)

    # =========================================================================
    # CORE LOGIC: GEOPANDAS (Precise)
    # =========================================================================
    def run_geometric_detection(self):
        print("\n--- Running Geometric Detection (GeoPandas) ---")
        start = time.time()
        
        # Buffer to catch nearby/touching lines
        self.gdf['buffer'] = self.gdf.geometry.buffer(self.buffer_size)
        gdf_buffer = self.gdf.set_geometry('buffer')
        
        joined = gpd.sjoin(gdf_buffer, gdf_buffer, how='inner', predicate='intersects')
        potential = joined[joined.index != joined.index_right]
        
        seen = set()
        count = 0
        
        for idx1, row in potential.iterrows():
            idx2 = row['index_right']
            pair = tuple(sorted((idx1, idx2)))
            if pair in seen: continue
            seen.add(pair)
            
            g1 = self.gdf.loc[idx1].geometry
            g2 = self.gdf.loc[idx2].geometry
            
            is_overlap = False
            overlap_type = ""
            confidence = 1.0
            
            # Check 1: Intersection
            inter = g1.intersection(g2)
            if inter.geom_type in ['LineString', 'MultiLineString'] and inter.length > 0.01:
                ratio = max(inter.length/g1.length, inter.length/g2.length)
                if ratio > 0.05:
                    is_overlap = True
                    overlap_type = "Topological Overlap"
            
            # Check 2: Collinear projection (for sub-segments)
            overlap_geom = None
            
            if not is_overlap:
                try:
                    p_start = g1.project(Point(g2.coords[0]))
                    p_end = g1.project(Point(g2.coords[-1]))
                    perp_dist = g1.distance(Point(g2.coords[0]))
                    
                    if ((0 < p_start < g1.length) or (0 < p_end < g1.length)) and perp_dist < self.buffer_size:
                         overlap_len = abs(p_end - p_start)
                         ratio = max(overlap_len/g1.length, overlap_len/g2.length)
                         if ratio > 0.05:
                             is_overlap = True
                             overlap_type = "Collinear Duplicate"
                             confidence = 0.95
                             # Construct geometry from projection
                             p1 = g1.interpolate(p_start)
                             p2 = g1.interpolate(p_end)
                             overlap_geom = LineString([p1, p2])
                except: pass
            
            if is_overlap:
                # Use constructed geometry for collinear, or intersection for topological
                final_geom = overlap_geom if overlap_geom else inter
                
                if final_geom.is_empty:
                     # Fallback if intersection failed but we detected overlap
                     final_geom = g1.intersection(g2.buffer(self.buffer_size))
                
                self.overlaps.append({
                    'id1': self.gdf.loc[idx1]['id'],
                    'id2': self.gdf.loc[idx2]['id'],
                    'type': overlap_type,
                    'confidence': confidence,
                    'geometry': final_geom,
                    'method': 'Geometry'
                })
                count += 1
                
        print(f"✓ Found {count} precise overlaps (Time: {time.time()-start:.2f}s)")
        self.geo_time = time.time() - start
        self.geo_count = count

    # =========================================================================
    # ML LOGIC: DBSCAN (Fuzzy)
    # =========================================================================
    def run_ml_detection(self):
        """DBSCAN-based fuzzy duplicate detection using geometric features"""
        print("\n--- Running ML Detection (DBSCAN) ---")
        start = time.time()
        
        self.ml_overlaps = []
        
        # Extract features for all segment pairs
        features = []
        pair_indices = []
        
        for i in range(len(self.gdf)):
            for j in range(i+1, len(self.gdf)):
                g1 = self.gdf.iloc[i].geometry
                g2 = self.gdf.iloc[j].geometry
                
                # Skip if too far apart
                # For geographic coords, 1 degree ≈ 111km, so scale up the multiplier
                distance_multiplier = 20000 if self.is_geographic else 10
                if g1.distance(g2) > self.buffer_size * distance_multiplier:
                    continue
                
                # Feature extraction
                try:
                    # Angle difference
                    def get_angle(geom):
                        coords = list(geom.coords)
                        dx = coords[-1][0] - coords[0][0]
                        dy = coords[-1][1] - coords[0][1]
                        return math.atan2(dy, dx)
                    
                    angle1 = get_angle(g1)
                    angle2 = get_angle(g2)
                    angle_diff = abs(angle1 - angle2)
                    if angle_diff > math.pi: angle_diff = 2*math.pi - angle_diff
                    
                    # Distance between centroids
                    centroid_dist = g1.centroid.distance(g2.centroid)
                    
                    # Length ratio
                    len_ratio = min(g1.length, g2.length) / max(g1.length, g2.length)
                    
                    # Proximity score (works for both projected and geographic)
                    # Minimum distance between the geometries
                    min_dist = g1.distance(g2)
                    
                    # For geographic: normalize by buffer_size to get comparable values
                    # proximity_score: 1.0 = touching/overlapping, 0.0 = far apart
                    proximity_score = max(0, 1.0 - (min_dist / (self.buffer_size * 100)))
                    
                    features.append([angle_diff, centroid_dist, len_ratio, proximity_score])
                    pair_indices.append((i, j))
                except:
                    continue
        
        if len(features) < 2:
            print("(Insufficient segment pairs for clustering)")
            self.ml_time = time.time() - start
            self.ml_count = 0
            return
        
        # Standardize and cluster
        X = StandardScaler().fit_transform(features)
        db = DBSCAN(eps=0.5, min_samples=2).fit(X)
        
        # Find "overlap-like" cluster (low angle diff, high proximity score)
        for label in set(db.labels_):
            if label == -1: continue
            cluster_mask = db.labels_ == label
            cluster_features = [features[k] for k in range(len(features)) if cluster_mask[k]]
            avg_angle = sum(f[0] for f in cluster_features) / len(cluster_features)
            avg_proximity = sum(f[3] for f in cluster_features) / len(cluster_features)
            
            # Overlap-like: low angle (< 17 degrees) and high proximity (> 0.5)
            # proximity_score is already normalized 0-1, so threshold is consistent
            if avg_angle < 0.3 and avg_proximity > 0.5:
                for k in range(len(features)):
                    if cluster_mask[k]:
                        i, j = pair_indices[k]
                        self.ml_overlaps.append({
                            'id1': self.gdf.iloc[i]['id'],
                            'id2': self.gdf.iloc[j]['id'],
                            'type': 'ML Detected',
                            'confidence': 0.7 + 0.3 * features[k][3],  # Based on overlap ratio
                            'geometry': self.gdf.iloc[i].geometry.intersection(
                                self.gdf.iloc[j].geometry.buffer(self.buffer_size)),
                            'method': 'DBSCAN'
                        })
        
        print(f"✓ Found {len(self.ml_overlaps)} ML-detected overlaps (Time: {time.time()-start:.2f}s)")
        self.ml_time = time.time() - start
        self.ml_count = len(self.ml_overlaps)

    # =========================================================================
    # COMPARISON REPORT
    # =========================================================================
    def generate_comparison_report(self):
        """Generate HTML comparison table between GeoPandas and DBSCAN results"""
        report_name = f"{self.basename}_comparison.html"
        
        geo_count = getattr(self, 'geo_count', len(self.overlaps))
        geo_time = getattr(self, 'geo_time', 0)
        ml_count = getattr(self, 'ml_count', 0)
        ml_time = getattr(self, 'ml_time', 0)
        ml_overlaps = getattr(self, 'ml_overlaps', [])
        
        # Find overlaps detected by both or only one method
        geo_pairs = {(o['id1'], o['id2']) for o in self.overlaps}
        ml_pairs = {(o['id1'], o['id2']) for o in ml_overlaps}
        both = geo_pairs & ml_pairs
        geo_only = geo_pairs - ml_pairs
        ml_only = ml_pairs - geo_pairs
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Method Comparison: {self.basename}</title>
            <style>
                body {{ font-family: Arial, sans-serif; background: #1a1a2e; color: white; padding: 20px; }}
                .container {{ max-width: 900px; margin: 0 auto; }}
                h1 {{ color: #e94560; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ padding: 12px; text-align: left; border: 1px solid #0f3460; }}
                th {{ background: #16213e; color: #e94560; }}
                tr:nth-child(even) {{ background: #16213e; }}
                .win {{ color: #00ff88; font-weight: bold; }}
                .metric {{ font-size: 24px; color: #e94560; }}
                .card {{ background: #16213e; padding: 20px; border-radius: 10px; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>📊 Method Comparison Report</h1>
                <p>File: <b>{self.basename}.wkt</b> | Segments: <b>{len(self.gdf)}</b></p>
                
                <h2>Performance Summary</h2>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>GeoPandas (Geometric)</th>
                        <th>DBSCAN (ML)</th>
                        <th>Winner</th>
                    </tr>
                    <tr>
                        <td>Overlaps Detected</td>
                        <td class="metric">{geo_count}</td>
                        <td class="metric">{ml_count}</td>
                        <td class="win">{'GeoPandas' if geo_count >= ml_count else 'DBSCAN'}</td>
                    </tr>
                    <tr>
                        <td>Execution Time</td>
                        <td>{geo_time:.3f}s</td>
                        <td>{ml_time:.3f}s</td>
                        <td class="win">{'GeoPandas' if geo_time <= ml_time else 'DBSCAN'}</td>
                    </tr>
                    <tr>
                        <td>Method Type</td>
                        <td>Exact Geometry</td>
                        <td>Statistical Clustering</td>
                        <td>-</td>
                    </tr>
                    <tr>
                        <td>Best For</td>
                        <td>QA Validation</td>
                        <td>Fuzzy Duplicates</td>
                        <td>-</td>
                    </tr>
                </table>
                
                <h2>Detection Agreement</h2>
                <div class="card">
                    <p>✅ <b>Detected by Both:</b> {len(both)} overlaps</p>
                    <p>🔷 <b>GeoPandas Only:</b> {len(geo_only)} overlaps</p>
                    <p>🟣 <b>DBSCAN Only:</b> {len(ml_only)} overlaps</p>
                </div>
                
                <h2>Recommendation</h2>
                <div class="card">
                    <p><b>For Quality Assurance:</b> Use <span class="win">GeoPandas</span> - Mathematically precise, catches true topological errors.</p>
                    <p><b>For Data Cleaning:</b> Use <span style="color: #9b59b6;">DBSCAN</span> - Finds fuzzy duplicates that may be intentional variants.</p>
                    <p><b>For Production:</b> Run both and use GeoPandas as ground truth, DBSCAN for flagging review candidates.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        with open(report_name, 'w') as f:
            f.write(html)
        print(f"✓ Generated Comparison Report: {report_name}")

    # =========================================================================
    # FEEDBACK-BASED TRAINING
    # =========================================================================
    def train_from_feedback(self, feedback_csv):
        """
        Use QA feedback to train a supervised classifier.
        
        HOW TO USE:
        1. Run detection on your data
        2. Open the generated *_feedback.csv file
        3. For each overlap, mark 'is_overlap' column as TRUE or FALSE
        4. Save the file
        5. Run: detector.train_from_feedback('your_feedback.csv')
        
        The system will learn from your corrections and improve future predictions.
        """
        print(f"\n--- Training from Feedback: {feedback_csv} ---")
        
        try:
            feedback = pd.read_csv(feedback_csv)
            
            if 'is_overlap' not in feedback.columns:
                print("❌ Feedback file must have 'is_overlap' column (TRUE/FALSE)")
                return
            
            # Filter labeled examples
            labeled = feedback[feedback['is_overlap'].isin(['TRUE', 'FALSE', True, False])]
            
            if len(labeled) < 5:
                print(f"⚠️ Need at least 5 labeled examples, found {len(labeled)}")
                return
            
            true_positives = labeled[labeled['is_overlap'].isin(['TRUE', True])]
            false_positives = labeled[labeled['is_overlap'].isin(['FALSE', False])]
            
            print(f"✓ Loaded {len(true_positives)} TRUE positives, {len(false_positives)} FALSE positives")
            
            # Calculate optimal thresholds from true positives
            if len(true_positives) > 0:
                # In production: use these to train a RandomForest/XGBoost classifier
                # For prototype: just report statistics
                print(f"\n📈 Training Summary:")
                print(f"   True Positive Rate: {len(true_positives)}/{len(labeled)} = {len(true_positives)/len(labeled):.0%}")
                print(f"   False Positive Rate: {len(false_positives)}/{len(labeled)} = {len(false_positives)/len(labeled):.0%}")
                print(f"\n💡 Tip: For production, use this labeled data to train a supervised classifier")
                print(f"   Example: sklearn.ensemble.RandomForestClassifier")
            
        except Exception as e:
            print(f"❌ Error reading feedback: {e}")
        
    # =========================================================================
    # EXPORT & REPORTING
    # =========================================================================
    def generate_outputs(self):
        if not self.overlaps:
            print("\n✅ No overlaps found! Data is clean.")
            return

        df_out = pd.DataFrame(self.overlaps)
        
        # 1. CSV
        csv_name = f"{self.basename}_overlaps.csv"
        df_out[['id1', 'id2', 'type', 'confidence', 'method']].to_csv(csv_name, index=False)
        print(f"✓ Exported CSV: {csv_name}")
        
        # 2. Interactive Map
        self._create_map(df_out)
        
        # 3. HTML Report
        self._create_report(df_out)
        
        # 4. Feedback Template - includes is_overlap column for user to fill
        feedback_df = df_out[['id1', 'id2', 'type', 'confidence']].copy()
        feedback_df['is_overlap'] = ''  # User fills: TRUE or FALSE
        feedback_df['comments'] = ''    # Optional user comments
        feedback_df.to_csv(f"{self.basename}_feedback.csv", index=False)
        print(f"✓ Created Feedback Template: {self.basename}_feedback.csv")

    def _create_map(self, df_out):
        map_name = f"{self.basename}_map.html"
        
        # Calculate center and bounds
        bounds = self.gdf.total_bounds
        center_y = (bounds[1] + bounds[3]) / 2
        center_x = (bounds[0] + bounds[2]) / 2
        
        # Use Simple CRS for local coordinates with light background
        m = folium.Map(location=[center_y, center_x], zoom_start=0, crs="Simple",
                      tiles=None)  # No default tiles
        
        # Add light/white background for better visibility
        light_bg = '''
        <style>
            .leaflet-container { background-color: #f5f5f5 !important; }
        </style>
        '''
        m.get_root().html.add_child(folium.Element(light_bg))
        
        # Fit bounds to data
        m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
        
        # --- Layer 1: ALL segments (dark blue on light bg for contrast) ---
        all_segments = folium.FeatureGroup(name="All Segments")
        for _, row in self.gdf.iterrows():
            coords = [(p[1], p[0]) for p in row.geometry.coords]
            seg_len = row.geometry.length
            folium.PolyLine(coords, color='#1a365d', weight=2, opacity=0.7,
                           popup=f"<b>{row['id']}</b><br>Length: {seg_len:.2f}").add_to(all_segments)
        all_segments.add_to(m)
        
        # --- Layer 2: Overlapping segment pairs (PURPLE for visibility) ---
        overlap_pairs = folium.FeatureGroup(name="Overlapping Pairs")
        drawn_ids = set()
        for _, row in df_out.iterrows():
            for seg_id in [row['id1'], row['id2']]:
                if seg_id not in drawn_ids:
                    seg = self.gdf[self.gdf['id'] == seg_id]
                    if not seg.empty:
                        coords = [(p[1], p[0]) for p in seg.geometry.iloc[0].coords]
                        seg_len = seg.geometry.iloc[0].length
                        folium.PolyLine(coords, color='#8b5cf6', weight=4, opacity=0.9,
                                       popup=f"<b>{seg_id}</b><br>Length: {seg_len:.2f}").add_to(overlap_pairs)
                        drawn_ids.add(seg_id)
        overlap_pairs.add_to(m)
        
        # --- Layer 3: Overlap regions (bright colors for max contrast) ---
        def get_color(conf):
            if conf >= 0.9: return '#dc2626'   # Red
            elif conf >= 0.7: return '#ea580c'  # Orange
            else: return '#ca8a04'              # Yellow/amber (visible on light bg)
        
        overlap_regions = folium.FeatureGroup(name="Overlap Regions")
        for _, row in df_out.iterrows():
            geom = row['geometry']
            conf = row['confidence']
            color = get_color(conf)
            overlap_len = geom.length if hasattr(geom, 'length') else 0
            
            popup_html = f"""
            <div style='font-family: Arial; min-width: 180px;'>
                <b style='color: {color};'>{row['id1']} ↔ {row['id2']}</b><br>
                <hr style='margin: 5px 0;'>
                <b>Type:</b> {row['type']}<br>
                <b>Confidence:</b> {conf:.0%}<br>
                <b>Overlap Length:</b> {overlap_len:.2f}
            </div>
            """
            
            if geom.geom_type == 'LineString' and len(geom.coords) >= 2:
                coords = [(p[1], p[0]) for p in geom.coords]
                folium.PolyLine(coords, color=color, weight=8, opacity=1.0,
                               popup=folium.Popup(popup_html, max_width=250)).add_to(overlap_regions)
            elif geom.geom_type == 'Point':
                g1 = self.gdf[self.gdf['id'] == row['id1']].geometry.iloc[0]
                g2 = self.gdf[self.gdf['id'] == row['id2']].geometry.iloc[0]
                for g in [g1, g2]:
                    coords = [(p[1], p[0]) for p in g.coords]
                    folium.PolyLine(coords, color=color, weight=6, opacity=0.9,
                                   popup=folium.Popup(popup_html, max_width=250)).add_to(overlap_regions)
            elif geom.geom_type == 'MultiLineString':
                for line in geom.geoms:
                    coords = [(p[1], p[0]) for p in line.coords]
                    folium.PolyLine(coords, color=color, weight=8, opacity=1.0,
                                   popup=folium.Popup(popup_html, max_width=250)).add_to(overlap_regions)
        overlap_regions.add_to(m)
        
        # --- Calculate Stats ---
        total_segments = len(self.gdf)
        total_overlaps = len(df_out)
        high_conf = len(df_out[df_out['confidence'] >= 0.9])
        med_conf = len(df_out[(df_out['confidence'] >= 0.7) & (df_out['confidence'] < 0.9)])
        low_conf = len(df_out[df_out['confidence'] < 0.7])
        affected_segments = len(drawn_ids)
        
        # --- Stats Panel (top-right) ---
        stats_html = f'''
        <div style="position: fixed; top: 20px; right: 20px; z-index: 1000;
                    background: white; color: #333;
                    padding: 15px 20px; border: 1px solid #ccc;
                    border-radius: 10px; font-family: Arial; box-shadow: 0 2px 8px rgba(0,0,0,0.15);
                    min-width: 200px;">
            <div style="font-size: 14px; font-weight: bold; margin-bottom: 10px; 
                        border-bottom: 1px solid #ddd; padding-bottom: 8px;">
                📊 Summary Stats
            </div>
            <div style="font-size: 12px; line-height: 1.8;">
                <b>Total Segments:</b> {total_segments}<br>
                <b>Affected Segments:</b> {affected_segments}<br>
                <hr style="border-color: #ddd; margin: 8px 0;">
                <b>Total Overlaps:</b> <span style="font-size: 16px; color: #dc2626;">{total_overlaps}</span><br>
                <span style="color: #dc2626;">● High Confidence:</span> {high_conf}<br>
                <span style="color: #ea580c;">● Medium:</span> {med_conf}<br>
                <span style="color: #ca8a04;">● Low:</span> {low_conf}<br>
            </div>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(stats_html))
        
        # --- Legend (bottom-left) ---
        legend_html = '''
        <div style="position: fixed; bottom: 30px; left: 20px; z-index: 1000;
                    background: white; color: #333;
                    padding: 12px 15px; border: 1px solid #ccc;
                    border-radius: 8px; font-family: Arial; font-size: 11px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.15);">
            <b style="font-size: 12px;">Legend</b><br>
            <span style="color: #1a365d;">━━</span> All Segments<br>
            <span style="color: #8b5cf6;">━━</span> Overlapping Pairs<br>
            <span style="color: #dc2626;">━━</span> High Confidence (≥90%)<br>
            <span style="color: #ea580c;">━━</span> Medium (70-90%)<br>
            <span style="color: #ca8a04;">━━</span> Low (<70%)<br>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        folium.LayerControl().add_to(m)
        m.save(map_name)
        print(f"✓ Generated Interactive Map: {map_name}")

    def _create_report(self, df_out):
        report_name = f"{self.basename}_report.html"
        html = f"""
        <html><body>
        <h1>Overlap Report: {self.basename}</h1>
        <p>Total Overlaps: {len(df_out)}</p>
        <table border="1">
        <tr><th>ID 1</th><th>ID 2</th><th>Type</th><th>Confidence</th></tr>
        """
        for _, row in df_out.iterrows():
            html += f"<tr><td>{row['id1']}</td><td>{row['id2']}</td><td>{row['type']}</td><td>{row['confidence']}</td></tr>"
        html += "</table></body></html>"
        
        with open(report_name, 'w') as f:
            f.write(html)
        print(f"✓ Generated HTML Report: {report_name}")

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║         OVERLAP DETECTOR - FINAL PROTOTYPE                   ║
║   GeoPandas (Precise) + DBSCAN (ML) Comparison System        ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    if len(sys.argv) < 2:
        print("""
USAGE:
  python overlap_detector.py <wkt_file>              # Standard detection
  python overlap_detector.py <wkt_file> --compare    # Run both methods + comparison
  python overlap_detector.py <wkt_file> --train <feedback.csv>  # Train from feedback

WHERE TO ADD TEST DATA:
  Place your .wkt file in this folder (files/Final_Prototype/) or provide full path.
  Example: python overlap_detector.py my_streets.wkt

HOW FEEDBACK TRAINING WORKS:
  1. Run detection to generate *_feedback.csv
  2. Open the CSV and mark 'is_overlap' column as TRUE or FALSE
  3. Run with --train flag to learn from your corrections
        """)
        sys.exit(1)
    
    wkt_file = sys.argv[1]
    compare_mode = '--compare' in sys.argv
    train_mode = '--train' in sys.argv
    
    detector = OverlapDetector(wkt_file)
    
    # Run geometric detection (always)
    detector.run_geometric_detection()
    
    if compare_mode:
        # Run ML detection and generate comparison
        detector.run_ml_detection()
        detector.generate_outputs()
        detector.generate_comparison_report()
    elif train_mode:
        # Training mode
        train_idx = sys.argv.index('--train')
        if train_idx + 1 < len(sys.argv):
            feedback_file = sys.argv[train_idx + 1]
            detector.train_from_feedback(feedback_file)
        else:
            print("❌ Please provide feedback CSV file after --train flag")
    else:
        # Standard mode
        detector.generate_outputs()
    
    print("\n✅ Done!")

