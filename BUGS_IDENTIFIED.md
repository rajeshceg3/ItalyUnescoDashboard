# Bugs Identified in UNESCO World Heritage Sites Dashboard

## Critical Issues

### 1. **Missing Latitude and Longitude Columns in CSV Data**
**Severity:** Critical
**File:** `data/unesco_sites_italy.csv` and scraper output
**Issue:** The CSV file only contains 6 columns: `Site Name`, `Image URL`, `Location`, `Year Listed`, `UNESCO Data`, `Description`. The application expects 8 columns including `Latitude` and `Longitude`.
**Impact:** 
- Map generation fails or shows incorrect locations
- All sites will use default coordinates (Rome)
- No meaningful geographical visualization

**Root Cause:** The scraper doesn't extract coordinate data from Wikipedia pages.

### 2. **Potential IndexError in show_site_details Function**
**Severity:** High
**File:** `app.py`, line 284
**Code:** `site_info_series = site_info_list.iloc[0]`
**Issue:** If `site_info_list` is empty, accessing `iloc[0]` will raise an IndexError.
**Impact:** Application crash when trying to view details of a non-existent site.

### 3. **Missing Error Handling for DataFrame Operations**
**Severity:** Medium
**File:** `app.py`, lines 124-128
**Issue:** The map center calculation doesn't handle cases where all coordinates are NaN after conversion.
**Code:**
```python
map_center = [df_map_data['Latitude'].mean(), df_map_data['Longitude'].mean()]
if pd.isna(map_center[0]) or pd.isna(map_center[1]):
    map_center = [DEFAULT_LATITUDE, DEFAULT_LONGITUDE]
```
**Impact:** If mean() returns NaN, the map might not render correctly.

### 4. **Inconsistent Return Types in update_country_data Function**
**Severity:** Medium  
**File:** `app.py`, lines 316-354
**Issue:** The function returns different dictionary structures (with and without certain keys) based on whether data exists.
**Impact:** Potential KeyError when accessing return values in Gradio outputs.

### 5. **Race Condition in Dependency Installation**
**Severity:** Medium
**File:** `app.py` and `scraper.py`
**Issue:** Both files attempt to install dependencies and modify PATH/PYTHONPATH simultaneously, which can lead to conflicts.
**Impact:** Installation failures or import errors in multi-process environments.

### 6. **Hardcoded Image URL Index in Scraper**
**Severity:** Medium
**File:** `scraper.py`, line 164
**Code:** `IMAGE_URL_INDEX = 1`
**Issue:** Assumes image is always in the second column of Wikipedia tables, which may not be true for all countries.
**Impact:** Incorrect or missing images for some countries.

### 7. **No Validation for Country Name Input**
**Severity:** Low
**File:** `scraper.py`, line 293
**Issue:** The scraper doesn't validate if the country name will produce a valid Wikipedia URL.
**Impact:** Silent failures for invalid country names.

### 8. **Potential Memory Leak in Map Generation**
**Severity:** Low
**File:** `app.py`, line 173
**Issue:** Folium maps are created repeatedly without proper cleanup.
**Impact:** Increased memory usage over time.

## Logic Issues

### 9. **Redundant Column Reordering**
**Severity:** Low
**File:** `app.py`, lines 104-105
**Issue:** Reordering columns to EXPECTED_COLUMNS even when columns might be missing.
**Impact:** Potential silent data corruption if CSV has fewer columns.

### 10. **Inefficient Search Implementation**
**Severity:** Low
**File:** `app.py`, lines 193-199
**Issue:** String conversion on every search even though data should already be strings.
**Impact:** Unnecessary performance overhead.

## Security Issues

### 11. **Subprocess Security Risk**
**Severity:** Medium
**File:** `app.py` and `scraper.py`
**Issue:** Using `subprocess.check_call` without proper input validation for pip installations.
**Impact:** Potential command injection if library names are maliciously crafted.

### 12. **Unvalidated URL Construction**
**Severity:** Low
**File:** `scraper.py`, line 98
**Issue:** Wikipedia URL is constructed without validation.
**Impact:** Potential for requesting invalid URLs.

## Performance Issues

### 13. **Inefficient Card Rendering**
**Severity:** Low
**File:** `app.py`, lines 175-214
**Issue:** Cards are re-rendered completely on every search instead of filtering existing ones.
**Impact:** Slow response times for large datasets.

### 14. **Repeated DataFrame Operations**
**Severity:** Low
**File:** `app.py`, various lines
**Issue:** Multiple `.copy()` operations on DataFrames without necessity.
**Impact:** Increased memory usage.

## Testing Issues

### 15. **Incomplete Test Coverage**
**Severity:** Medium
**File:** `tests/test_scraper_execution.py`
**Issue:** Only tests scraper execution, doesn't test the main application functionality.
**Impact:** No automated testing for critical bugs in the web application.

### 16. **No Coordinate Validation in Tests**
**Severity:** Medium
**File:** `tests/test_scraper_execution.py`
**Issue:** Tests don't verify that coordinate columns are present in scraped data.
**Impact:** The critical coordinate bug went undetected.

## Documentation Issues

### 17. **Misleading README Instructions**
**Severity:** Low
**File:** `README.md`
**Issue:** Claims automatic dependency installation works reliably, but it has known issues.
**Impact:** User frustration when setup fails.

---

## Summary

**Total Bugs Found:** 17
- Critical: 1
- High: 1  
- Medium: 7
- Low: 8

The most critical issue is the missing coordinate data, which renders the core mapping functionality useless. This should be addressed immediately by fixing the scraper to extract latitude and longitude information from Wikipedia or alternative sources.