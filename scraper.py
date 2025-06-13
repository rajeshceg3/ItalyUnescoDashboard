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
            print(f"Installing {lib_name} using pip install --user...")
            try:
                # Consistently use --user and ensure PATH is updated for user-installed packages
                user_site_packages = subprocess.check_output(
                    [sys.executable, "-m", "site", "--user-site"],
                    text=True,
                    stderr=subprocess.STDOUT # Capture stderr as well
                ).strip()

                user_bin_path = os.path.expanduser("~/.local/bin")

                # Ensure user bin path is in PATH
                if user_bin_path not in os.environ["PATH"]:
                    print(f"Adding {user_bin_path} to PATH")
                    os.environ["PATH"] = f"{user_bin_path}:{os.environ['PATH']}"

                # Ensure user site packages is in PYTHONPATH
                if "PYTHONPATH" not in os.environ:
                    os.environ["PYTHONPATH"] = user_site_packages
                elif user_site_packages not in os.environ["PYTHONPATH"]:
                    os.environ["PYTHONPATH"] = f"{user_site_packages}:{os.environ['PYTHONPATH']}"

                print(f"Current PATH: {os.environ.get('PATH', 'Not set')}")
                print(f"Current PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")

                subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", lib_name])

                # Re-check import after attempting to fix environment
                # For subprocess calls, environment changes in os.environ might not propagate directly
                # depending on how subprocess is used. However, for direct imports later in *this* script,
                # sys.path should be updated by site.py based on PYTHONPATH.
                # We might need to sys.path.append(user_site_packages) if direct import still fails.
                if user_site_packages not in sys.path:
                    print(f"Adding {user_site_packages} to sys.path")
                    sys.path.append(user_site_packages)

                importlib.import_module(module_name) # Try importing again
                print(f"Successfully installed and imported {lib_name}.")

            except subprocess.CalledProcessError as e:
                print(f"Error installing {lib_name}: {e}")
                output = e.output.decode('utf-8') if e.output else "No output"
                print(f"Pip install output:\n{output}")
                all_installed = False
            except ImportError as ie:
                print(f"Failed to import {lib_name} even after attempting installation and PATH/PYTHONPATH updates.")
                print(f"Current sys.path: {sys.path}")
                print(f"ImportError: {ie}")
                all_installed = False
            except Exception as ex:
                print(f"An unexpected error occurred during installation of {lib_name}: {ex}")
                all_installed = False


    if not all_installed:
        print("One or more libraries could not be installed. Please check the errors above and the environment settings.")
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
        # Initialize fields with default "N/A"
        site_name, image_url, location, year_listed, unesco_data, description = ["N/A"] * 6

        try:
            cells = row.find_all(["th", "td"])

            # General check for cell count
            if len(cells) < 1:
                # print(f"Warning: Row {row_index} has no cells. Skipping.") # Warnings commented
                continue # Skip this iteration if no cells

            # Site Name (cells[0])
            if len(cells) > 0:
                site_name = cells[0].get_text(strip=True)
            # else: site_name remains "N/A"

            # Image URL (cells[1])
            if len(cells) > 1:
                img_tag = cells[1].find("img")
                if img_tag and img_tag.has_attr("src"):
                    image_url_temp = img_tag["src"]
                    if image_url_temp.startswith("//"):
                        image_url = "https:" + image_url_temp
                    elif image_url_temp.startswith("/"):
                        image_url = "https://en.wikipedia.org" + image_url_temp
                    elif image_url_temp.startswith("http"):
                        image_url = image_url_temp
            # else: image_url remains "N/A"

            # Location (cells[2])
            if len(cells) > 2:
                location = cells[2].get_text(strip=True)
            # else: location remains "N/A"

            # Year Listed (cells[3])
            if len(cells) > 3:
                year_listed = cells[3].get_text(strip=True)
            # else: year_listed remains "N/A"

            # UNESCO Data (cells[4])
            if len(cells) > 4:
                unesco_data = cells[4].get_text(strip=True)
            # else: unesco_data remains "N/A"

            # Description (cells[5])
            if len(cells) > 5:
                description = cells[5].get_text(strip=True)
            # else: description remains "N/A"

            if len(cells) < 6 and len(cells) > 0 :
                 # print(f"Note: Row {row_index} ('{site_name}') has {len(cells)} cells (expected 6). Some data fields were defaulted to 'N/A'.") # Warnings commented
                 pass


        except Exception as e:
            # print(f"Error processing cells for row {row_index} ('{site_name}'): {e}. Data for this row might be incomplete or defaulted.") # Warnings commented
            # Fields already initialized to "N/A", so they will retain that if an error occurs.
            pass

        sites_data.append([
            site_name, image_url, location, year_listed, unesco_data, description
        ])

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
