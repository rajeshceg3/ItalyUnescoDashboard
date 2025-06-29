# UNESCO World Heritage Sites Dashboard

This project displays UNESCO World Heritage Sites using a web interface. It scrapes data from Wikipedia for various countries, allows users to select a country, view its sites on a map, search for specific sites, and see detailed information for each site.

## Features

*   **Multi-Country Support:** Allows selection and viewing of UNESCO sites from different countries (initially supporting Italy and France).
*   **Data Scraping:** Automatically scrapes data of UNESCO World Heritage Sites from Wikipedia for user-specified countries (e.g., Italy, France).
*   **Interactive Map:** Displays sites on a Folium-powered interactive map for the selected country.
*   **Card View:** Shows sites in an organized card layout.
*   **Search Functionality:** Allows users to search for sites by name or description within the selected country's dataset.
*   **Detailed View:** Provides a detailed information page for each site.
*   **In-App Guidance:** Instructs users on how to generate data if it's missing for a selected country.
*   **Automatic Dependency Installation:** Both the scraper and the application attempt to install required Python libraries automatically.

## Technology Stack

*   **Python:** Core programming language.
*   **Gradio:** For creating the web application interface.
*   **Pandas:** For data manipulation and creating DataFrames.
*   **Folium:** For generating the interactive map.
*   **Beautiful Soup 4 (bs4):** For parsing HTML content (used in `scraper.py`).
*   **Requests:** For making HTTP requests to fetch web content (used in `scraper.py`).
*   **lxml:** XML and HTML parser (used by BeautifulSoup).

## Setup and Usage

1.  **Clone the Repository:**
    ```bash
    git clone <repository_url> # Replace <repository_url> with the actual URL
    cd <repository_directory> # Replace <repository_directory> with the name of the cloned directory
    ```

2.  **Dependencies:**
    Both the scraper (`scraper.py`) and the application (`app.py`) are designed to automatically install their required Python libraries when first run. This includes libraries like `pandas`, `gradio`, `folium`, `beautifulsoup4`, and `requests`. The scripts use `pip install --user` for this purpose. Ensure you have Python and pip installed.

3.  **Run the Scraper (Optional but Recommended):**
    To ensure you have the latest data, or if the data file for a specific country (e.g., `data/unesco_sites_italy.csv`, `data/unesco_sites_france.csv`) is missing, run the scraper for that country:
    ```bash
    python scraper.py --country "Italy"
    python scraper.py --country "France"
    # Replace "France" with any other country you wish to scrape
    ```
    This will create or update the respective country-specific CSV file in the `data/` directory.
    The application will also provide instructions if data for a selected country is not found.

4.  **Run the Application:**
    Once the desired data files are present (e.g., at least for Italy, which loads by default), you can start the Gradio web application:
    ```bash
    python app.py
    ```
    The application will typically open in your default web browser. If not, the console output will provide a URL (usually `http://127.0.0.1:7860`) to access the dashboard.

## Data Source

The data for the UNESCO World Heritage Sites is scraped from Wikipedia (e.g., `https://en.wikipedia.org/wiki/List_of_World_Heritage_Sites_in_COUNTRY_NAME`). The scraper is designed to construct the correct URL based on the country name provided.

The scraped data is processed and stored locally in country-specific CSV files in the `data/` directory (e.g., `data/unesco_sites_italy.csv`, `data/unesco_sites_france.csv`). If a file for a selected country is not present when `app.py` is run, you should run `scraper.py` for that country, or follow the in-app instructions.

## Contributing

Contributions are welcome! If you have suggestions for improvements, new features, or find any bugs, please feel free to:

1.  Fork the repository.
2.  Create a new branch for your changes.
3.  Make your changes and commit them.
4.  Submit a pull request for review.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
