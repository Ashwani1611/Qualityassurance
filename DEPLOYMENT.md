# Overlap Detection Web App - Replit Deployment Guide

## Quick Setup on Replit

### 1. Create New Repl
1. Go to [replit.com](https://replit.com)
2. Click "Create Repl"
3. Select "Python" as the template
4. Name it "overlap-detector"

### 2. Upload Files
Copy these files from `files/Final_Prototype/` to your Repl:
- `app.py` (main Streamlit app)
- `overlap_detector.py` (core detection logic)
- `requirements.txt` (dependencies)
- `.replit` (Replit configuration)
- `sample_data.wkt` (optional: test data)
- `map_data.wkt` (optional: test data)

### 3. Run the App
1. Click the "Run" button
2. Wait for dependencies to install (first run takes ~2 minutes)
3. Access your app at the URL shown (format: `https://overlap-detector.username.repl.co`)

## Features Available

| Feature | Description |
|---------|-------------|
| **File Upload** | Drag-and-drop WKT files |
| **Sample Data** | Pre-loaded test datasets |
| **Test Generator** | Create synthetic overlaps |
| **GeoPandas** | Precise geometric detection |
| **DBSCAN** | ML-based fuzzy detection |
| **Interactive Map** | Dark-themed Folium visualization |
| **Comparison** | Side-by-side method analysis |
| **Feedback** | Upload corrected results for training |
| **Export** | Download CSV, HTML, maps |

## Usage Flow

```
1. Upload/Select Data
   ↓
2. Choose Method (Geo/ML/Both)
   ↓
3. Run Detection
   ↓
4. View Results (Map/Table)
   ↓
5. Export or Train
```

## Troubleshooting

### App won't start
- Check requirements.txt installed correctly
- Verify all files are uploaded
- Restart the Repl

### Map not displaying
- Ensure Streamlit version is 1.28.0+
- Check browser console for errors

### File upload errors
- Check WKT format (LINESTRING only)
- File size limit: ~10MB on free Replit

## Customization

### Change theme colors
Edit CSS in `app.py` lines 25-40

### Add more sample data
Place `.wkt` files in the main directory

### Modify detection parameters
Edit `BUFFER_SIZE` in `overlap_detector.py`

## Sharing Your App

1. Click "Publish" in Replit
2. Share the generated URL
3. Users can interact without login

## Local Testing (Before Replit)

```bash
cd files/Final_Prototype
pip install -r requirements.txt
streamlit run app.py
```

Open browser to `http://localhost:8501`

## Support

If you encounter issues:
1. Check Replit logs (Console tab)
2. Verify file paths are relative
3. Ensure Python version is 3.9+
