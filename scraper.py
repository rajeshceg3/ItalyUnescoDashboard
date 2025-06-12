import os
import subprocess
import sys
import importlib

def install_libraries():
    """Installs necessary Python libraries and verifies installation."""
    libraries = ["requests", "beautifulsoup4", "pandas", "lxml"]
    all_installed = True
    for lib_name in libraries:
        # Standardize module name for import check (e.g., beautifulsoup4 -> bs4)
        module_name = lib_name
        if lib_name == "beautifulsoup4":
            module_name = "bs4"

        try:
            importlib.import_module(module_name)
            print(f"{lib_name} is already installed.")
        except ImportError:
            print(f"Installing {lib_name}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", lib_name])
                # Verify installation by trying to import again
                importlib.import_module(module_name)
                print(f"Successfully installed {lib_name}.")
            except subprocess.CalledProcessError as e:
                print(f"Error installing {lib_name}: {e}")
                all_installed = False
            except ImportError:
                print(f"Failed to import {lib_name} even after attempting installation.")
                all_installed = False

    if not all_installed:
        print("One or more libraries could not be installed. Please check the errors above.")
        sys.exit(1)

# Call install_libraries at the beginning of the script execution
install_libraries()

# Import necessary libraries after installation check
import requests
from bs4 import BeautifulSoup
import pandas as pd

def scrape_unesco_sites():
    """
    Scrapes UNESCO World Heritage Sites in Italy from Wikipedia.
    """
    # Create data directory if it doesn't exist
    if not os.path.exists("data"):
        os.makedirs("data")
        print("Created 'data' directory.")

    # Fetch HTML content
    url = "https://en.wikipedia.org/wiki/List_of_World_Heritage_Sites_in_Italy"
    print(f"Fetching data from {url}...")
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        print("Successfully fetched HTML content.")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL: {e}")
        return

    # Parse HTML
    print("Parsing HTML content...")
    soup = BeautifulSoup(response.content, "lxml")

    # Find the main table
    caption_text = "World Heritage Sites"
    table = None

    # Try finding table by caption first
    for caption_tag in soup.find_all("caption"):
        if caption_text in caption_tag.get_text():
            table = caption_tag.find_parent("table")
            if table:
                print("Found table by caption: 'World Heritage Sites'")
                break

    if not table:
        # Fallback: try to find table by common class if caption method fails
        print("Caption method failed. Trying to find table by class 'wikitable sortable'...")
        table = soup.find("table", class_="wikitable sortable")
        if table:
            print("Found table by class 'wikitable sortable'.")

    if not table:
        print("Error: Could not find the World Heritage Sites table after trying multiple methods.")
        return

    # Extract data
    print("Extracting data from table...")
    sites_data = []
    # The first row after headers is usually index 1, but if the table structure is complex
    # (e.g. multiple header rows), we might need to adjust.
    # Let's inspect the first few rows to be sure.
    # For now, we assume the first row [0] is headers, data starts from [1].
    rows = table.find_all("tr")
    if len(rows) < 2:
        print("Error: Table has no data rows.")
        return

    for row_index, row in enumerate(rows[1:], start=1):  # Skip header row (usually rows[0])
        try:
            cells = row.find_all(["th", "td"])

            if len(cells) < 6:
                print(f"Warning: Skipping row {row_index} due to insufficient cells (found {len(cells)}, expected at least 6): {row.get_text(strip=True)[:70]}...")
                continue

            site_name = cells[0].get_text(strip=True)

            # Image URL
            img_tag = cells[1].find("img")
            image_url = "N/A" # Default if no image
            if img_tag and img_tag.has_attr("src"):
                image_url_temp = img_tag["src"]
                if image_url_temp.startswith("//"):
                    image_url = "https:" + image_url_temp
                elif image_url_temp.startswith("/"): # Should not happen with Wikipedia's current setup but good to have
                     image_url = "https://en.wikipedia.org" + image_url_temp
                elif image_url_temp.startswith("http"): # Already a full URL
                    image_url = image_url_temp
                # else, it might be a relative path not starting with / - less common for src attributes.

            location = cells[2].get_text(strip=True)
            year_listed = cells[3].get_text(strip=True)
            unesco_data = cells[4].get_text(strip=True)
            description = cells[5].get_text(strip=True)

            sites_data.append([
                site_name, image_url, location, year_listed, unesco_data, description
            ])
        except IndexError as e:
            print(f"Warning: Skipping row {row_index} due to missing cells (IndexError): {row.get_text(strip=True)[:70]}... Error: {e}")
        except Exception as e:
            print(f"Warning: Error processing row {row_index}: {row.get_text(strip=True)[:70]}... Error: {e}")

    if not sites_data:
        print("No data extracted. The table might be empty or the structure is not as expected.")
        return

    # Store data in DataFrame
    print(f"Extracted data for {len(sites_data)} sites. Creating DataFrame...")
    df = pd.DataFrame(sites_data, columns=[
        "Site Name", "Image URL", "Location", "Year Listed", "UNESCO Data", "Description"
    ])

    # Save DataFrame to CSV
    csv_path = os.path.join("data", "unesco_sites_italy.csv")
    try:
        df.to_csv(csv_path, index=False)
        print(f"Successfully scraped {len(df)} sites.")
        print(f"Data saved to {csv_path}")
    except Exception as e:
        print(f"Error saving DataFrame to CSV: {e}")


if __name__ == "__main__":
    scrape_unesco_sites()
