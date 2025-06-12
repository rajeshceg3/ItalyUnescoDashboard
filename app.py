import os
import subprocess
import sys
import importlib
from functools import partial
import pandas as pd
import gradio as gr

# --- Dependency Installation ---
def install_dependencies():
    """Installs necessary libraries if not already present."""
    libraries = ["gradio", "pandas", "folium"]
    all_installed = True
    for lib_name in libraries:
        try:
            importlib.import_module(lib_name)
            print(f"{lib_name} is already installed.")
        except ImportError:
            print(f"Installing {lib_name}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", lib_name])
                importlib.import_module(lib_name) # Verify installation
                print(f"Successfully installed {lib_name}.")
            except subprocess.CalledProcessError as e:
                print(f"Error installing {lib_name}: {e}")
                all_installed = False
            except ImportError:
                print(f"Failed to import {lib_name} even after attempting installation.")
                all_installed = False

    if not all_installed:
        print("One or more essential libraries could not be installed. Exiting.")
        sys.exit(1)

install_dependencies()
import folium # Import after installation check

# --- Constants and Data Loading ---
DATA_FILE = "data/unesco_sites_italy.csv"
DEFAULT_LATITUDE = 41.8719  # Rome
DEFAULT_LONGITUDE = 12.5674

def load_data():
    """Loads data from CSV and adds dummy coordinates if needed."""
    try:
        if os.path.exists(DATA_FILE):
            df = pd.read_csv(DATA_FILE)
            if 'Latitude' not in df.columns:
                df['Latitude'] = DEFAULT_LATITUDE
            if 'Longitude' not in df.columns:
                df['Longitude'] = DEFAULT_LONGITUDE
            # Ensure no NaN in coordinates for Folium
            df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce').fillna(DEFAULT_LATITUDE)
            df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce').fillna(DEFAULT_LONGITUDE)

            # Fill NA for other potentially missing text fields to avoid errors in display
            for col in ['Site Name', 'Image URL', 'Location', 'Year Listed', 'UNESCO Data', 'Description']:
                if col in df.columns:
                    df[col] = df[col].fillna("N/A")
                else:
                    df[col] = "N/A" # Add column if missing
            return df
        else:
            print(f"Error: Data file '{DATA_FILE}' not found. Please run scraper.py first.")
            # Return an empty DataFrame with expected columns to prevent errors later
            return pd.DataFrame(columns=['Site Name', 'Image URL', 'Location', 'Year Listed', 'UNESCO Data', 'Description', 'Latitude', 'Longitude'])
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame(columns=['Site Name', 'Image URL', 'Location', 'Year Listed', 'UNESCO Data', 'Description', 'Latitude', 'Longitude'])

initial_df = load_data()

# --- Map Generation ---
def generate_map_html(df_map_data):
    """Generates an HTML representation of a Folium map with markers for sites."""
    if df_map_data.empty or not all(col in df_map_data.columns for col in ['Latitude', 'Longitude', 'Site Name']):
        return "<p style='text-align:center; color:grey;'>Map data is unavailable or incomplete. Cannot render map.</p>"

    # Central point for the map (can be mean of coordinates or a fixed point)
    map_center = [df_map_data['Latitude'].mean(), df_map_data['Longitude'].mean()]

    # Handle potential NaN values that might have slipped through, or if all coordinates are default
    if pd.isna(map_center[0]) or pd.isna(map_center[1]):
        map_center = [DEFAULT_LATITUDE, DEFAULT_LONGITUDE]

    site_map = folium.Map(location=map_center, zoom_start=5)

    for _, row in df_map_data.iterrows():
        try:
            lat = float(row['Latitude'])
            lon = float(row['Longitude'])
            if pd.isna(lat) or pd.isna(lon):
                print(f"Skipping site {row['Site Name']} due to invalid coordinates.")
                continue
            folium.Marker(
                location=[lat, lon],
                popup=f"{row['Site Name']}<br><a href='{row.get('Image URL', '#')}' target='_blank'>View Image</a>", # Basic popup
                tooltip=row['Site Name']
            ).add_to(site_map)
        except ValueError as ve:
            print(f"Coordinate conversion error for site {row['Site Name']}: {ve}. Using default for map display if possible.")
        except Exception as e:
            print(f"Error adding marker for site {row['Site Name']}: {e}")

    return site_map._repr_html_()

# --- Gradio UI Definition ---
with gr.Blocks(css="custom.css", title="Italian UNESCO World Heritage Sites") as app:
    # --- State Variables ---
    all_sites_df_state = gr.State(initial_df)
    filtered_sites_df_state = gr.State(initial_df.copy())
    # selected_site_name_state = gr.State() # Not strictly needed if show_site_details directly updates components

    # --- Main Content Area (Visible by default) ---
    with gr.Column(elem_id="main_content_area_dynamic") as main_content_area:
        gr.Markdown("# Italian UNESCO World Heritage Sites Dashboard", elem_id="main_title_md_dynamic") # Changed elem_id
        search_box = gr.Textbox(label="Search by Name or Description", elem_id="search_box_dynamic") # Changed elem_id
        gr.Markdown("## Map of Sites", elem_id="map_title_md_dynamic") # Changed elem_id
        map_output = gr.HTML(elem_id="map_output_html_dynamic") # Changed elem_id
        sites_cards_area = gr.Column(elem_id="sites_cards_area_dynamic") # This will hold the dynamic cards

    # --- Detailed View Area (Hidden by default) ---
    with gr.Column(visible=False, elem_id="detailed_view_area_dynamic") as detailed_view_area: # Changed elem_id
        back_to_list_btn = gr.Button("⬅️ Back to List", elem_id="back_to_list_btn_dynamic") # Changed elem_id
        detail_site_name_md = gr.Markdown(elem_id="detail_site_name_md_dynamic") # Changed elem_id
        detail_image_display = gr.Image(show_label=False, interactive=False, height=300, elem_id="detail_image_display_dynamic") # Changed elem_id
        detail_desc_text = gr.Textbox(label="Description", interactive=False, lines=5, elem_id="detail_desc_text_dynamic") # Changed elem_id
        detail_location_text = gr.Textbox(label="Location (Provinces)", interactive=False, elem_id="detail_location_text_dynamic") # Changed elem_id
        detail_year_text = gr.Textbox(label="Year Listed", interactive=False, elem_id="detail_year_text_dynamic") # Changed elem_id
        detail_unesco_text = gr.Textbox(label="UNESCO Data", interactive=False, elem_id="detail_unesco_text_dynamic") # Changed elem_id

    # --- Card Rendering Function ---
    def render_site_cards(df_render_data, all_sites_for_button_action):
        card_list = []
        if df_render_data.empty:
            return [gr.Markdown("No sites found matching your criteria.", elem_id="no_sites_found_md")]

        for index, row_data in df_render_data.iterrows():
            site_name = row_data['Site Name']
            with gr.Column(elem_classes=['site-card']) as card:
                if pd.notna(row_data['Image URL']) and row_data['Image URL'] != "N/A":
                    gr.Image(value=row_data['Image URL'], show_label=False, interactive=False, height=200)
                else:
                    gr.Image(value=None, show_label=False, interactive=False, height=200) # Placeholder or hide
                gr.Markdown(f"### {site_name}", elem_classes=['card-title'])
                gr.Textbox(value=row_data['Location'], label="Location", interactive=False, lines=1)

                # Use functools.partial to pass site_name to the click handler
                # The handler 'show_site_details' will need to know which components to update.
                # It will return a dictionary of component: gr.update(...)
                view_details_btn = gr.Button("View Details")
                view_details_btn.click(
                    fn=partial(show_site_details, site_name), # Pass site_name directly
                    inputs=[all_sites_for_button_action],      # Pass the full dataset to find the site
                    outputs=[
                        main_content_area, detailed_view_area,
                        detail_site_name_md, detail_image_display, detail_desc_text,
                        detail_location_text, detail_year_text, detail_unesco_text
                    ]
                )
            card_list.append(card)
        return card_list

    # --- Event Handlers ---
    def update_search_results(query, all_sites_data_val):
        if not query:
            filtered_df = all_sites_data_val
        else:
            q_lower = query.lower()
            filtered_df = all_sites_data_val[
                all_sites_data_val['Site Name'].str.lower().str.contains(q_lower) |
                all_sites_data_val['Description'].str.lower().str.contains(q_lower)
            ]

        new_cards = render_site_cards(filtered_df, all_sites_df_state) # Pass state for button actions
        new_map_html = generate_map_html(filtered_df)

        # The output for sites_cards_area should be the list of components directly
        return new_cards, new_map_html, filtered_df # Update cards, map, and filtered_df_state

    def show_site_details(site_name_to_display, all_sites_data_val):
        """Handles click on 'View Details' button."""
        site_info_series = all_sites_data_val[all_sites_data_val['Site Name'] == site_name_to_display].iloc[0]

        return {
            main_content_area: gr.update(visible=False),
            detailed_view_area: gr.update(visible=True),
            detail_site_name_md: gr.update(value=f"## {site_info_series['Site Name']}"),
            detail_image_display: gr.update(value=site_info_series['Image URL'] if pd.notna(site_info_series['Image URL']) and site_info_series['Image URL'] != "N/A" else None),
            detail_desc_text: gr.update(value=site_info_series['Description']),
            detail_location_text: gr.update(value=site_info_series['Location']),
            detail_year_text: gr.update(value=site_info_series['Year Listed']),
            detail_unesco_text: gr.update(value=site_info_series['UNESCO Data'])
        }

    def back_to_list_view_fn():
        """Handles click on 'Back to List' button."""
        return {
            main_content_area: gr.update(visible=True),
            detailed_view_area: gr.update(visible=False)
        }

    # --- Connect Event Handlers ---
    search_box.submit(
        fn=update_search_results,
        inputs=[search_box, all_sites_df_state],
        outputs=[sites_cards_area, map_output, filtered_sites_df_state]
    )
    # Optional: Trigger search on text change (can be resource-intensive)
    # search_box.change(
    #     fn=update_search_results,
    #     inputs=[search_box, all_sites_df_state],
    #     outputs=[sites_cards_area, map_output, filtered_sites_df_state]
    # )

    back_to_list_btn.click(
        fn=back_to_list_view_fn,
        inputs=[],
        outputs=[main_content_area, detailed_view_area]
    )

    # --- App Load Action ---
    def initial_load(all_sites_data_val):
        # Same as update_search_results but with empty query
        filtered_df = all_sites_data_val
        initial_cards = render_site_cards(filtered_df, all_sites_df_state)
        initial_map_html = generate_map_html(filtered_df)
        return initial_cards, initial_map_html, filtered_df

    app.load(
        fn=initial_load,
        inputs=[all_sites_df_state],
        outputs=[sites_cards_area, map_output, filtered_sites_df_state]
    )

if __name__ == "__main__":
    if initial_df.empty and not os.path.exists(DATA_FILE):
        print(f"CRITICAL: Data file {DATA_FILE} not found. The app will launch but will be mostly empty.")
        print("Please ensure 'scraper.py' has been run successfully.")
    elif initial_df.empty:
        print(f"WARNING: Data file {DATA_FILE} was found but appears to be empty or failed to load correctly.")
    print("Launching Gradio app...")
    app.launch()
