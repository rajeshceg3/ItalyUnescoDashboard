import os
import subprocess
import sys
import importlib
import site # Added site module

def install_libraries():
    """Installs necessary Python libraries and verifies installation."""
    print("Consider using a Python virtual environment for managing dependencies.")
    libraries = ["requests", "beautifulsoup4", "pandas", "lxml"]
    
    # Validate library names for security
    import re
    valid_name_pattern = r'^[a-zA-Z0-9_\-\.]+$'
    for lib in libraries:
        if not re.match(valid_name_pattern, lib):
            print(f"Error: Invalid library name '{lib}'. Skipping for security reasons.")
            continue
    
    all_installed = True

    # Get user site packages path once
    user_site_dir = site.getusersitepackages()
    if user_site_dir: # Can be a string or a list of strings
        if isinstance(user_site_dir, list):
            user_site_dir = user_site_dir[0] # Take the first one
        if user_site_dir not in sys.path:
            print(f"Adding user site packages directory {user_site_dir} to sys.path.")
            sys.path.append(user_site_dir)
    else:
        print("Warning: Could not determine user site packages directory. Pip installs might not be found without shell restart.")

    for lib_name in libraries:
        module_name = lib_name
        if lib_name == "beautifulsoup4":
            module_name = "bs4"

        try:
            importlib.import_module(module_name)
            print(f"{lib_name} (as {module_name}) is already installed.")
        except ImportError:
            print(f"Attempting to install {lib_name} using pip install --user...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", lib_name])
                print(f"Successfully ran pip install for {lib_name}.")

                importlib.invalidate_caches() # Invalidate caches before re-import

                importlib.import_module(module_name) # Try importing again
                print(f"Successfully imported {lib_name} (as {module_name}) after installation.")

            except subprocess.CalledProcessError as e:
                print(f"Error during 'pip install --user {lib_name}': {e}")
                output = e.output.decode('utf-8') if e.output and hasattr(e.output, 'decode') else "No output captured"
                print(f"Pip install output for {lib_name}:\n{output}")
                print(f"Please try installing {lib_name} manually using 'pip install --user {lib_name}' and restart the script.")
                all_installed = False
            except ImportError as ie:
                print(f"Failed to import {lib_name} (as {module_name}) even after attempting installation.")
                print(f"ImportError: {ie}")
                print(f"Current sys.path: {sys.path}")
                print(f"If you recently installed this package, you might need to restart the script or your environment.")
                all_installed = False
            except Exception as ex:
                print(f"An unexpected error occurred during the installation or import of {lib_name}: {ex}")
                all_installed = False

    if not all_installed:
        print("\nOne or more essential libraries could not be installed or imported correctly.")
        print("Please review the error messages above and ensure the libraries are installed in a location accessible by Python.")
        sys.exit(1)

# Call install_libraries at the beginning of the script execution
install_libraries()

# Import necessary libraries after installation check
import argparse # Added for command-line arguments
import requests
from bs4 import BeautifulSoup
import pandas as pd
# sys is already imported at the top

def scrape_unesco_sites(country_name):
    """
    Scrapes UNESCO World Heritage Sites for a specific country from Wikipedia.
    """
    # Create data directory if it doesn't exist
    # Using exist_ok=True is cleaner than checking os.path.exists
    os.makedirs("data", exist_ok=True)
    print("Ensured 'data' directory exists.")

    # Construct URL and filename dynamically
    country_name_url_part = country_name.replace(' ', '_')
    url = f"https://en.wikipedia.org/wiki/List_of_World_Heritage_Sites_in_{country_name_url_part}"

    # Validate URL format
    import re
    url_pattern = r'^https://en\.wikipedia\.org/wiki/List_of_World_Heritage_Sites_in_[a-zA-Z0-9_\-\.]+$'
    if not re.match(url_pattern, url):
        print(f"Error: Generated URL format is invalid: {url}")
        return

    print(f"Fetching data for {country_name} from {url}...")
    try:
        response = requests.get(url, timeout=30)  # Add timeout
        response.raise_for_status()  # Raise an exception for bad status codes
        print(f"Successfully fetched HTML content for {country_name}.")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL for {country_name}: {e}")
        print(f"Please verify that the Wikipedia page exists: {url}")
        return

    # Parse HTML
    print(f"Parsing HTML content for {country_name}...")
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
    rows = table.find_all("tr")
    if len(rows) < 2: # Need at least one header row and one data row
        print("Error: Table has insufficient rows (must have at least a header and a data row).")
        return

    # Dynamic Column Indexing
    headers_row = rows[0].find_all('th')
    header_texts = [h.get_text(strip=True).lower() for h in headers_row]

    column_map = {}
    # Define canonical names and possible header variations
    # Image URL will be handled specially as it's often not a text header.
    # For Wikipedia, image is consistently in the second column (index 1).
    expected_cols = {
        "site_name": ["site name", "name", "site", "official name"],
        "location": ["location", "region", "province"], # Removed coordinates from here
        "year_listed": ["year", "listed", "inscription date", "date of inscription"],
        "unesco_data": ["criteria", "unesco data", "id", "reference"], # UNESCO ID or criteria
        "description": ["description", "summary", "notes", "brief description"],
        "coordinates": ["coordinates", "coord", "lat", "lon", "latitude", "longitude"]
    }

    for canonical_name, possible_headers in expected_cols.items():
        found = False
        for i, header_text in enumerate(header_texts):
            for p_header in possible_headers:
                if p_header in header_text:
                    column_map[canonical_name] = i
                    found = True
                    break
            if found:
                break
        if not found:
            print(f"Warning: Could not find header for '{canonical_name}'. This column will be missing or defaulted to 'N/A'.")
            column_map[canonical_name] = -1 # Indicate missing header with -1 or some other sentinel

    # Check for crucial missing headers
    crucial_headers = ["site_name", "location"] # Example: site name is essential
    for ch in crucial_headers:
        if column_map.get(ch, -1) == -1:
            print(f"Error: Crucial header '{ch}' is missing from the table. Cannot reliably process data. Exiting.")
            return

    # Image URL is typically in the second column (index 1) and doesn't have a text header.
    # We will keep its handling based on a fixed index, but add robustness.
    IMAGE_URL_INDEX = 1


    # Import geocoding function
    def extract_coordinates_from_location(location_text):
        """Extract coordinates from location text or return defaults."""
        import re
        # Look for coordinate patterns in the location text
        coord_pattern = r'(\d+\.?\d*)[°\s]*[NS][\s,]*(\d+\.?\d*)[°\s]*[EW]'
        match = re.search(coord_pattern, location_text)
        if match:
            lat, lon = match.groups()
            return float(lat), float(lon)
        
        # Try to extract decimal coordinates
        decimal_pattern = r'(-?\d+\.?\d+)[,\s]+(-?\d+\.?\d+)'
        match = re.search(decimal_pattern, location_text)
        if match:
            lat, lon = match.groups()
            return float(lat), float(lon)
            
        return None, None

    def get_coordinates_for_site(site_name, location_text):
        """Get coordinates for a site, trying multiple approaches."""
        # First try to extract from location text
        lat, lon = extract_coordinates_from_location(location_text)
        if lat is not None and lon is not None:
            return lat, lon
            
        # Default coordinates for common locations (this is a simple fallback)
        location_coords = {
            # Italy defaults by region
            'rome': (41.9028, 12.4964),
            'milan': (45.4642, 9.1900),
            'florence': (43.7696, 11.2558),
            'venice': (45.4408, 12.3155),
            'naples': (40.8518, 14.2681),
            'turin': (45.0703, 7.6869),
            'sicily': (37.5999, 14.0154),
            'sardinia': (40.1209, 9.0129),
            # Add more as needed
        }
        
        location_lower = location_text.lower()
        for city, coords in location_coords.items():
            if city in location_lower:
                return coords
                
        # Return None if no coordinates found
        return None, None

    for row_index, row in enumerate(rows[1:], start=1):  # Skip header row
        # Initialize fields with default "N/A"
        site_name, image_url, location, year_listed, unesco_data, description = ["N/A"] * 6
        latitude, longitude = None, None

        cells = row.find_all(["td", "th"]) # Cells can be td or th in data rows sometimes

        # Helper function to get cell data safely
        def get_cell_data(column_name, cell_list):
            idx = column_map.get(column_name, -1)
            if idx != -1 and idx < len(cell_list):
                return cell_list[idx].get_text(strip=True)
            return "N/A"

        try:
            # Site Name
            site_name = get_cell_data("site_name", cells)

            # Image URL (special handling - index based)
            if IMAGE_URL_INDEX < len(cells):
                img_tag = cells[IMAGE_URL_INDEX].find("img")
                if img_tag and img_tag.has_attr("src"):
                    image_url_temp = img_tag["src"]
                    if image_url_temp.startswith("//"):
                        image_url = "https:" + image_url_temp
                    elif image_url_temp.startswith("/"):
                        image_url = "https://en.wikipedia.org" + image_url_temp
                    elif image_url_temp.startswith("http"):
                        image_url = image_url_temp
                    else:
                        image_url = "N/A" # Default if format is unexpected
                else:
                    image_url = "N/A" # Default if no img_tag or src
            else:
                image_url = "N/A" # Default if cell index is out of bounds

            # Location
            location = get_cell_data("location", cells)

            # Year Listed
            year_listed = get_cell_data("year_listed", cells)

            # UNESCO Data
            unesco_data = get_cell_data("unesco_data", cells)

            # Description
            description = get_cell_data("description", cells)

            # Extract Coordinates
            latitude, longitude = get_coordinates_for_site(site_name, location)
            
            # Logging for rows with potentially missing data based on expected columns
            missing_data_details = []
            if site_name == "N/A" and column_map.get("site_name", -1) != -1 : missing_data_details.append("Site Name")
            if location == "N/A" and column_map.get("location", -1) != -1 : missing_data_details.append("Location")
            if latitude is None or longitude is None: missing_data_details.append("Coordinates")
            # Add more checks if needed for other important fields

            if missing_data_details:
                print(f"Warning: Row {row_index} (Site: '{site_name if site_name != 'N/A' else 'Unknown'}') - Data for '{', '.join(missing_data_details)}' defaulted to 'N/A' due to missing cell or content.")

        except (AttributeError, IndexError) as e:
            current_site_name_for_log = site_name if site_name != "N/A" else "Unknown"
            print(f"Error processing cells for row {row_index} (Site: '{current_site_name_for_log}'): {e}. This might be due to an unexpected table structure. Data for this row set to 'N/A'.")
            site_name, image_url, location, year_listed, unesco_data, description = ["N/A"] * 6
            latitude, longitude = None, None

        # Use default coordinates if none found
        if latitude is None or longitude is None:
            latitude, longitude = 41.8719, 12.5674  # Default to Rome

        sites_data.append([
            site_name, image_url, location, year_listed, unesco_data, description, latitude, longitude
        ])

    if not sites_data:
        print("No data extracted. The table might be empty, or the structure after header processing is not as expected.")
        return

    # Store data in DataFrame
    print(f"Extracted data for {len(sites_data)} sites. Creating DataFrame...")
    df = pd.DataFrame(sites_data, columns=[
        "Site Name", "Image URL", "Location", "Year Listed", "UNESCO Data", "Description", "Latitude", "Longitude"
    ])

    # Save DataFrame to CSV
    country_name_file_part = country_name.lower().replace(' ', '_')
    csv_path = os.path.join("data", f"unesco_sites_{country_name_file_part}.csv")
    try:
        df.to_csv(csv_path, index=False)
        print(f"Successfully scraped {len(df)} sites for {country_name}.")
        print(f"Data saved to {csv_path}")
    except IOError as e:
        print(f"Error saving DataFrame to CSV for {country_name}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape UNESCO World Heritage Sites for a specific country.")
    parser.add_argument(
        "--country",
        type=str,
        default="Italy",
        help="Name of the country to scrape (e.g., 'France', 'Germany'). Defaults to 'Italy'."
    )
    args = parser.parse_args()

    country_input = args.country.strip()
    if not country_input:
        print("Error: Country name cannot be empty.")
        sys.exit(1)
    
    # Basic validation for country name
    import re
    if not re.match(r'^[a-zA-Z\s\-\'\.]+$', country_input):
        print("Error: Country name contains invalid characters. Only letters, spaces, hyphens, apostrophes, and periods are allowed.")
        sys.exit(1)
    
    if len(country_input) > 100:
        print("Error: Country name is too long (maximum 100 characters).")
        sys.exit(1)

    print(f"Starting scrape for country: {country_input}")
    scrape_unesco_sites(country_input)
