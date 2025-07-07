# Bugs Fixed

1.  **`scraper.py`: Overly Broad Exception Handling**
    *   **Description:** The scraper used a generic `except Exception as e`, which could hide specific errors.
    *   **Fix:** Replaced the generic exception with specific handlers for `requests.exceptions.RequestException` and `IOError`.

2.  **`app.py`: Incorrect DataFrame Indexing**
    *   **Description:** The `/api/sites/<int:site_id>` endpoint used `sites[site_id]` to access a row, which is not the correct way to select a row by its integer position in a pandas DataFrame.
    *   **Fix:** Created a new Flask API endpoint that uses `sites.iloc[site_id]` to correctly select a row by its integer position.

3.  **`app.py`: Missing Error Handling for Out-of-Bounds Access**
    *   **Description:** When a `site_id` was requested that was outside the valid range of the DataFrame's rows, the application would throw an `IndexError`, resulting in a 500 Internal Server Error.
    *   **Fix:** The new Flask API endpoint now checks if the `site_id` is within the valid range of the DataFrame's rows and returns a 404 Not Found error if it is not.
