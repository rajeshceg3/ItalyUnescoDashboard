# Italian UNESCO World Heritage Sites Dashboard

This project displays Italian UNESCO World Heritage Sites using a web interface. It scrapes data from Wikipedia, allows users to view sites on a map, search for specific sites, and see detailed information for each site.

## Features

*   **Data Scraping:** Automatically scrapes data of Italian UNESCO World Heritage Sites from Wikipedia.
*   **Interactive Map:** Displays sites on a Folium-powered interactive map.
*   **Card View:** Shows sites in an organized card layout.
*   **Search Functionality:** Allows users to search for sites by name or description.
*   **Detailed View:** Provides a detailed information page for each site, including image, description, location, year listed, and UNESCO data.
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
    To ensure you have the latest data, or if the `data/unesco_sites_italy.csv` file is missing, run the scraper first:
    ```bash
    python scraper.py
    ```
    This will create/update the `data/unesco_sites_italy.csv` file.

4.  **Run the Application:**
    Once the data file is present, you can start the Gradio web application:
    ```bash
    python app.py
    ```
    The application will typically open in your default web browser. If not, the console output will provide a URL (usually `http://127.0.0.1:7860`) to access the dashboard.

## Data Source

The data for the UNESCO World Heritage Sites in Italy is scraped from the Wikipedia page: [List of World Heritage Sites in Italy](https://en.wikipedia.org/wiki/List_of_World_Heritage_Sites_in_Italy).

The scraped data is processed and stored locally in the `data/unesco_sites_italy.csv` file. If this file is not present when `app.py` is run, or if you wish to refresh the data, you should run `scraper.py`.

## Contributing

Contributions are welcome! If you have suggestions for improvements, new features, or find any bugs, please feel free to:

1.  Fork the repository.
2.  Create a new branch for your changes.
3.  Make your changes and commit them.
4.  Submit a pull request for review.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
