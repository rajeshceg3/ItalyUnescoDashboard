import os
import sys
from functools import partial
import pandas as pd
import gradio as gr
from flask import Flask, jsonify
import folium
import json
import tempfile

# --- Flask App Initialization ---
app_flask = Flask(__name__)

# --- Constants and Data Loading ---
# DATA_FILE is now dynamically generated in load_data
DEFAULT_LATITUDE = 41.8719  # Default fallback Rome
DEFAULT_LONGITUDE = 12.5674
EXPECTED_COLUMNS = ['Site Name', 'Image URL', 'Location', 'Year Listed', 'UNESCO Data', 'Description', 'Latitude', 'Longitude']

def load_data(country_name="Italy"):
    """Loads data from CSV for a given country and adds dummy coordinates if needed."""
    data_file_path = os.path.join("data", f"unesco_sites_{country_name.lower().replace(' ', '_')}.csv")
    print(f"Attempting to load data for {country_name} from: {data_file_path}")

    if not os.path.exists(data_file_path):
        print(f"Error: Data file '{data_file_path}' not found for {country_name}. Scraper might need to be run for this country.")
        return pd.DataFrame(columns=EXPECTED_COLUMNS)

    try:
        df = pd.read_csv(data_file_path)

        # Ensure all expected columns exist, fill with N/A or default if not
        for col in EXPECTED_COLUMNS:
            if col not in df.columns:
                if col == 'Latitude':
                    df[col] = DEFAULT_LATITUDE
                elif col == 'Longitude':
                    df[col] = DEFAULT_LONGITUDE
                else:
                    df[col] = "N/A"

        # Ensure no NaN in coordinates for Folium
        df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce').fillna(DEFAULT_LATITUDE)
        df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce').fillna(DEFAULT_LONGITUDE)

        # Fill NA for other text fields
        # Note: Latitude and Longitude are already handled and are numeric.
        text_cols = [col for col in EXPECTED_COLUMNS if col not in ['Latitude', 'Longitude']]
        for col in text_cols:
            if col in df.columns: # Check if column exists before fillna
                df[col] = df[col].fillna("N/A")
            # If col was not in df.columns, it was already created and filled with "N/A" or default above.

        # Reorder columns to EXPECTED_COLUMNS order just in case CSV has different order
        df = df.reindex(columns=EXPECTED_COLUMNS)
        print(f"Successfully loaded and processed data for {country_name}.")
        return df
    except Exception as e:
        # This catches errors from pd.read_csv() or any processing steps above
        print(f"Error processing data for {country_name} from {data_file_path}: {e}")
        return pd.DataFrame(columns=EXPECTED_COLUMNS)

initial_df = load_data() # Loads "Italy" by default

@app_flask.route('/api/sites/<int:site_id>')
def get_site(site_id):
    """API endpoint to get site details by ID."""
    if site_id < 0 or site_id >= len(initial_df):
        return jsonify({"error": "Site not found"}), 404
    site_data = initial_df.iloc[site_id].to_dict()
    return jsonify(site_data)

# --- Map Generation ---
def generate_map_html(df_map_data, route_data=None):
    """
    Generates an HTML representation of a Folium map with markers for sites.
    If route_data is provided, it draws the route on the map.
    """
    if df_map_data.empty or not all(col in df_map_data.columns for col in ['Latitude', 'Longitude', 'Site Name']):
        return "<p style='text-align:center; color:grey;'>Map data is unavailable or incomplete. Cannot render map.</p>"

    # Central point for the map
    try:
        lat_mean = df_map_data['Latitude'].mean()
        lon_mean = df_map_data['Longitude'].mean()
        
        if pd.isna(lat_mean) or pd.isna(lon_mean) or lat_mean == 0 or lon_mean == 0:
            map_center = [DEFAULT_LATITUDE, DEFAULT_LONGITUDE]
        else:
            map_center = [lat_mean, lon_mean]
    except Exception as e:
        print(f"Error calculating map center: {e}")
        map_center = [DEFAULT_LATITUDE, DEFAULT_LONGITUDE]

    site_map = folium.Map(location=map_center, zoom_start=5, tiles='CartoDB positron')

    # Draw Route if available
    if route_data:
        points = []
        for i, site in enumerate(route_data):
            try:
                lat = float(site['Latitude'])
                lon = float(site['Longitude'])
                points.append([lat, lon])

                # Add ordered marker
                folium.Marker(
                    location=[lat, lon],
                    popup=folium.Popup(f"<b>#{i+1}: {site['Site Name']}</b>", max_width=300),
                    tooltip=f"{i+1}. {site['Site Name']}",
                    icon=folium.Icon(color='red', icon='info-sign', prefix='fa')
                ).add_to(site_map)
            except Exception:
                continue

        if len(points) > 1:
            folium.PolyLine(
                locations=points,
                color='blue',
                weight=5,
                opacity=0.7
            ).add_to(site_map)

    # Draw standard markers if no route (or in addition? Let's stick to just route markers if route is active to avoid clutter)
    if not route_data:
        for _, row in df_map_data.iterrows():
            try:
                lat = float(row['Latitude'])
                lon = float(row['Longitude'])
                if pd.isna(lat) or pd.isna(lon):
                    continue

                site_name = row.get('Site Name', 'N/A')
                image_url = row.get('Image URL', '#')
                location_text = row.get('Location', 'N/A')
                year_listed = row.get('Year Listed', 'N/A')

                popup_html = f"<b>{site_name}</b><br>"
                if image_url != '#' and image_url != 'N/A' and pd.notna(image_url):
                    popup_html += f"<a href='{image_url}' target='_blank'>View Image</a><br>"
                else:
                    popup_html += "No image available<br>"

                if location_text != 'N/A' and pd.notna(location_text):
                    popup_html += f"<b>Location:</b> {location_text}<br>"
                if year_listed != 'N/A' and pd.notna(year_listed):
                    popup_html += f"<b>Year Listed:</b> {year_listed}"

                if popup_html.endswith("<br>"):
                     popup_html = popup_html[:-4]

                folium.Marker(
                    location=[lat, lon],
                    popup=folium.Popup(popup_html, max_width=300),
                    tooltip=site_name
                ).add_to(site_map)
            except Exception as e:
                print(f"Error adding marker for site {row['Site Name']}: {e}")

    return site_map._repr_html_()

# --- Planning Logic Import ---
try:
    from planning import RoutePlanner, StrategicSortiePlanner, MissionSimulator
    from tactical import ThreatZone
    from assets import ASSET_REGISTRY
    from wargame import MonteCarloEngine
except ImportError as e:
    print(f"Warning: planning.py, tactical.py, assets.py or wargame.py not found. Route optimization features will be disabled. Error: {e}")
    RoutePlanner = None
    StrategicSortiePlanner = None
    ThreatZone = None
    ASSET_REGISTRY = {}
    MissionSimulator = None
    MonteCarloEngine = None


# --- Gradio UI Definition ---
# Note: In Gradio 6.0+, 'css' and 'title' moved to app.launch(), but 'title' still often works in Blocks.
# We will assume modern usage where we pass custom_css to Blocks if supported, or via launch.
# However, standard practice is still Blocks(css=...) in many versions.
with gr.Blocks(css="custom.css", title="Italian UNESCO World Heritage Sites") as app:
    # --- Fonts & Global Styles ---
    gr.HTML("""
        <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700&family=Open+Sans:wght@400;600&display=swap" rel="stylesheet">
    """)

    # --- State Variables ---
    all_sites_df_state = gr.State(initial_df)
    filtered_sites_df_state = gr.State(initial_df.copy())
    threats_state = gr.State([]) # List of ThreatZone objects
    mission_sorties_state = gr.State([]) # Store planned sorties for simulation
    wargame_results_state = gr.State({}) # Store COAs

    # --- Header ---
    main_title_md = gr.Markdown("# Italy UNESCO World Heritage Sites Dashboard", elem_id="main_title_md_dynamic")

    # --- Define Detail View Handler (defined early so it can be called) ---
    def show_site_details(site_name_to_display, current_all_sites_df):
        """Handles click on 'View Details' button."""
        site_row = current_all_sites_df[current_all_sites_df['Site Name'] == site_name_to_display]

        if site_row.empty:
            return (
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(value="## Site Not Found"),
                gr.update(value=None),
                gr.update(value="The requested site is not available in the current dataset."),
                gr.update(value="N/A"),
                gr.update(value="N/A"),
                gr.update(value="N/A")
            )

        site_info_series = site_row.iloc[0]

        image_url_val = site_info_series.get('Image URL', "N/A")
        image_to_display = None if pd.isna(image_url_val) or image_url_val == "N/A" else image_url_val

        return (
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(value=f"## {site_info_series.get('Site Name', 'N/A')}"),
            gr.update(value=image_to_display),
            gr.update(value=site_info_series.get('Description', 'N/A')),
            gr.update(value=site_info_series.get('Location', 'N/A')),
            gr.update(value=site_info_series.get('Year Listed', 'N/A')),
            gr.update(value=site_info_series.get('UNESCO Data', 'N/A'))
        )

    with gr.Tabs():
        # --- TAB 1: EXPLORER ---
        with gr.Tab("Site Explorer"):
            with gr.Column(elem_id="main_content_area_dynamic") as main_content_area:
                countries = ["Italy", "France"]
                country_dropdown = gr.Dropdown(
                    label="Select Country",
                    choices=countries,
                    value="Italy",
                    elem_id="country_dropdown_dynamic"
                )

                search_box = gr.Textbox(label="Search by Name or Description", elem_id="search_box_dynamic")
                gr.Markdown("## Map of Sites", elem_id="map_title_md_dynamic")
                map_output = gr.HTML(elem_id="map_output_html_dynamic")

                # Render Area for Cards - Moved decorator INSIDE the scope
                with gr.Column(elem_id="sites_cards_area_dynamic") as sites_cards_area:
                     # --- Card Rendering Function (using @gr.render) ---
                    @gr.render(inputs=[filtered_sites_df_state, all_sites_df_state], triggers=[app.load, search_box.submit, country_dropdown.change])
                    def render_site_cards(df_render_data, all_sites_df_state_for_click_handler):
                        if df_render_data.empty:
                            gr.Markdown("No sites found matching your criteria.", elem_id="no_sites_found_md")
                            return

                        for index, row_data in df_render_data.iterrows():
                            site_name = row_data['Site Name']
                            with gr.Group():
                                if pd.notna(row_data['Image URL']) and row_data['Image URL'] != "N/A":
                                    gr.HTML(f"<img src='{row_data['Image URL']}' style='height:200px; width:100%; object-fit:cover; border-radius: 8px;'>")
                                else:
                                    gr.HTML("<div style='height:200px; width:100%; background-color: #f0f0f0; display: flex; align-items: center; justify-content: center; border-radius: 8px; color: #888;'>No Image</div>")
                                gr.Markdown(f"### {site_name}", elem_classes=['card-title'])
                                gr.Textbox(value=row_data.get('Location', 'N/A'), label="Location", interactive=False, lines=1)

                                view_details_btn = gr.Button("View Details")

                                view_details_btn.click(
                                    fn=show_site_details,
                                    inputs=[gr.State(value=site_name), all_sites_df_state],
                                    outputs=[
                                        detailed_view_area,
                                        main_content_area,
                                        detail_site_name_md,
                                        detail_image_display,
                                        detail_desc_text,
                                        detail_location_text,
                                        detail_year_text,
                                        detail_unesco_text
                                    ]
                                )

                scrape_data_button = gr.Button("Generate/Refresh Data for Selected Country", visible=False, elem_id="scrape_data_button_dynamic")
                scraper_instructions_md = gr.Markdown("", visible=False, elem_id="scraper_instructions_md_dynamic")

            # --- Detailed View (Hidden by default, nested here to swap with main content) ---
            with gr.Column(visible=False, elem_id="detailed_view_area_dynamic") as detailed_view_area:
                back_to_list_btn = gr.Button("⬅️ Back to List", elem_id="back_to_list_btn_dynamic")
                detail_site_name_md = gr.Markdown(elem_id="detail_site_name_md_dynamic")
                detail_image_display = gr.Image(show_label=False, interactive=False, height=300, elem_id="detail_image_display_dynamic")
                detail_desc_text = gr.Textbox(label="Description", interactive=False, lines=5, elem_id="detail_desc_text_dynamic")
                detail_location_text = gr.Textbox(label="Location (Provinces)", interactive=False, elem_id="detail_location_text_dynamic")
                detail_year_text = gr.Textbox(label="Year Listed", interactive=False, elem_id="detail_year_text_dynamic")
                detail_unesco_text = gr.Textbox(label="UNESCO Data", interactive=False, elem_id="detail_unesco_text_dynamic")

        # --- TAB 2: MISSION PLANNER ---
        with gr.Tab("Mission Planner"):
            gr.Markdown("## Heritage Ops: Tactical Site Reconnaissance & Route Planning")
            gr.Markdown("Select a squad of sites to visit and generate an optimized tactical route.")

            with gr.Accordion("⚠️ Tactical Intel & Threat Assessment", open=False):
                with gr.Row():
                    with gr.Column():
                        threat_lat = gr.Number(label="Latitude", value=42.0)
                        threat_lon = gr.Number(label="Longitude", value=12.5)
                        threat_rad = gr.Slider(minimum=1, maximum=100, value=20, label="Radius (km)")
                        threat_name = gr.Textbox(label="Threat ID", value="Hostile Area")
                        add_threat_btn = gr.Button("Add Threat Zone")
                    with gr.Column():
                        threat_list_display = gr.JSON(label="Active Threats", value=[])
                        clear_threats_btn = gr.Button("Clear All Threats", variant="stop")

            with gr.Row():
                with gr.Column(scale=1):
                    # Initial choices from Italy (default)
                    site_names = sorted(initial_df['Site Name'].tolist()) if not initial_df.empty else []

                    mp_site_selector = gr.Dropdown(
                        choices=site_names,
                        label="Select Target Sites",
                        multiselect=True,
                        info="Select at least 2 sites."
                    )

                    mp_start_selector = gr.Dropdown(
                        choices=site_names,
                        label="Insertion Point (Start Site)",
                        info="Optional. If empty, the first selected site is used."
                    )

                    gr.Markdown("### Fleet Composition")
                    with gr.Row():
                         # Dynamic fleet composition
                         # For simplicity, we create specific inputs for each asset type count
                         fleet_uav_count = gr.Number(label="UAV (Drone)", value=1, precision=0, minimum=0)
                         fleet_suv_count = gr.Number(label="SUV (Ground)", value=0, precision=0, minimum=0)
                         fleet_helo_count = gr.Number(label="Helo (Air)", value=0, precision=0, minimum=0)
                         fleet_sedan_count = gr.Number(label="Sedan (Covert)", value=0, precision=0, minimum=0)

                    mp_execute_btn = gr.Button("Execute Mission Plan", variant="primary")
                    mp_download_file = gr.File(label="Download Orders")

                with gr.Column(scale=2):
                    mp_map_output = gr.HTML(label="Tactical Map")

            with gr.Row():
                 mp_sim_slider = gr.Slider(minimum=0, maximum=100, value=0, label="Mission Simulation Progress (%)", interactive=True)
                 mp_telemetry_output = gr.JSON(label="Live Telemetry", value={"status": "Standby"}, elem_id="mp_telemetry_output")

            with gr.Row():
                mp_briefing_output = gr.Markdown(label="Operational Briefing")

        # --- TAB 3: WAR ROOM (NEW) ---
        with gr.Tab("War Room"):
            gr.Markdown("## Predictive Operational Wargaming Engine (POWE)")
            gr.Markdown("Run Monte Carlo simulations to stress-test mission plans against probabilistic threats and environmental variables.")

            with gr.Row():
                wr_generate_btn = gr.Button("Generate Strategy Variants (COAs)", variant="primary")
                wr_simulate_btn = gr.Button("Run Monte Carlo Simulation (100 Runs)", variant="secondary")

            gr.Markdown("### Strategy Comparison")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Option A: SPEED")
                    wr_stats_speed = gr.JSON(label="Simulation Stats")
                    wr_select_speed_btn = gr.Button("Select Option A")
                with gr.Column():
                    gr.Markdown("#### Option B: STEALTH")
                    wr_stats_stealth = gr.JSON(label="Simulation Stats")
                    wr_select_stealth_btn = gr.Button("Select Option B")
                with gr.Column():
                    gr.Markdown("#### Option C: EFFICIENCY")
                    wr_stats_efficiency = gr.JSON(label="Simulation Stats")
                    wr_select_efficiency_btn = gr.Button("Select Option C")

            wr_plot_output = gr.Plot(label="Risk vs Reward Analysis")

    # --- HELPER: Update Mission Planner Choices ---
    def update_mp_choices(df):
        if df.empty:
            return gr.update(choices=[], value=[]), gr.update(choices=[], value=None)

        # Ensure site names are strings and filter out NaNs
        names = df['Site Name'].dropna().astype(str).tolist()
        names = sorted(list(set(names))) # Deduplicate and sort

        return gr.update(choices=names, value=[]), gr.update(choices=names, value=None)

    # --- HELPER: Threat Management ---
    def add_threat(lat, lon, rad, name, current_threats):
        if ThreatZone:
            new_threat = ThreatZone(lat, lon, rad, name)
            current_threats.append(new_threat)
        return [t.to_dict() for t in current_threats], current_threats

    def clear_threats():
        return [], []

    add_threat_btn.click(
        fn=add_threat,
        inputs=[threat_lat, threat_lon, threat_rad, threat_name, threats_state],
        outputs=[threat_list_display, threats_state]
    )

    clear_threats_btn.click(
        fn=clear_threats,
        inputs=[],
        outputs=[threat_list_display, threats_state]
    )

    # --- HELPER: Generate Mission Plan ---
    def generate_mission_plan(selected_sites, start_site_name, n_uav, n_suv, n_helo, n_sedan, all_sites_df, threats, strategy='speed'):
        if not RoutePlanner or not StrategicSortiePlanner:
            return "Error: Planning module not loaded.", "<p>Planning module missing.</p>", None, []

        if not selected_sites or len(selected_sites) < 2:
            return "## Mission Aborted\n\nPlease select at least 2 sites to form a route.", "", None, []

        # Construct Fleet safely
        fleet = []
        try:
            for _ in range(int(n_uav)): fleet.append(ASSET_REGISTRY.get("Drone (UAV)"))
            for _ in range(int(n_suv)): fleet.append(ASSET_REGISTRY.get("Ground Team (SUV)"))
            for _ in range(int(n_helo)): fleet.append(ASSET_REGISTRY.get("Air Support (Helo)"))
            for _ in range(int(n_sedan)): fleet.append(ASSET_REGISTRY.get("Covert Ops (Sedan)"))
        except Exception:
            pass # Handle safely if int conversion fails or None is fetched

        # Remove None values if ASSET_REGISTRY lookup failed
        fleet = [f for f in fleet if f is not None]

        if not fleet:
             return "## Mission Aborted\n\nPlease allocate at least one valid asset to the fleet.", "", None, []

        # Filter dataframe for selected sites
        selected_df = all_sites_df[all_sites_df['Site Name'].isin(selected_sites)]

        if selected_df.empty:
            return "## Error\n\nSelected sites not found in database.", "", None, []

        # Determine start site
        if start_site_name and start_site_name in selected_sites:
            start_row = selected_df[selected_df['Site Name'] == start_site_name].iloc[0]
        else:
            start_row = selected_df.iloc[0] # Default to first found

        start_site_dict = start_row.to_dict()

        target_sites = []
        for _, row in selected_df.iterrows():
            if row['Site Name'] != start_site_dict['Site Name']:
                target_sites.append(row.to_dict())

        # If no other sites, just return
        if not target_sites:
             return "## Error\n\nNo target sites selected (only base selected).", "", None, []

        planner = StrategicSortiePlanner(start_site_dict, target_sites)

        # Use the new fleet mission planner with strategy
        sorties = planner.plan_fleet_mission(fleet=fleet, threats=threats, strategy=strategy)

        # --- Generate Map ---
        map_html = generate_multi_sortie_map(all_sites_df, sorties, start_site_dict, threats)

        # --- Generate Briefing ---
        briefing = f"# 📜 Strategic Operational Briefing ({strategy.upper()})\n\n"
        briefing += f"**Base of Operations:** {start_site_dict['Site Name']}\n"
        briefing += f"**Fleet Strength:** {len(fleet)} Assets\n"
        briefing += f"**Total Sorties Generated:** {len(sorties)}\n\n"

        if threats:
            briefing += f"**⚠️ Active Threat Zones:** {len(threats)}\n\n"

        total_mission_dist = 0
        total_mission_time = 0
        total_fuel = 0

        for sortie in sorties:
            status_icon = "🟢" if sortie['status'] == "GREEN" else "🔴"

            briefing += f"### {status_icon} Sortie #{sortie['id']} ({sortie['asset_name']}) - {sortie['status_msg']}\n"
            briefing += f"- **Distance:** {sortie['distance']} km\n"
            briefing += f"- **Est. Duration:** {sortie['est_duration_hrs']} hrs\n"
            briefing += f"- **Fuel Est:** {sortie['fuel_used']} units\n"
            briefing += f"- **Risk Score:** {sortie['risk_score']}\n"
            briefing += f"- **Targets:** {sortie['site_count']}\n\n"

            briefing += "**Itinerary:**\n"
            for i, site in enumerate(sortie['route']):
                 briefing += f"{i+1}. {site['Site Name']}\n"
            briefing += "\n---\n"

            total_mission_dist += sortie['distance']
            total_mission_time += sortie['est_duration_hrs']
            total_fuel += sortie['fuel_used']

        briefing += f"### 📊 Mission Totals\n"
        briefing += f"**Total Distance:** {round(total_mission_dist, 2)} km\n"
        briefing += f"**Total Fuel Required:** {round(total_fuel, 1)} units\n"
        briefing += f"**Total Operational Time:** {round(total_mission_time, 2)} hrs\n"

        mission_data = {
            "mission_name": "Heritage Ops Recon",
            "base": start_site_dict['Site Name'],
            "sorties": [{k:v for k,v in s.items() if k != 'detailed_path'} for s in sorties],
            "threats": [t.to_dict() for t in threats] if threats else []
        }
        mission_json = json.dumps(mission_data, indent=4, default=str)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            tmp_file.write(mission_json)
            mission_file_path = tmp_file.name

        return briefing, map_html, mission_file_path, sorties

    def generate_multi_sortie_map(df_map_data, sorties, start_site, threats=None, simulation_states=None):
        """
        Generates map, optionally adding markers for simulation states (current positions).
        """
        if df_map_data.empty:
             return "<p>No data</p>"

        try:
            lat_mean = df_map_data['Latitude'].mean()
            lon_mean = df_map_data['Longitude'].mean()
            map_center = [lat_mean, lon_mean]
        except:
            map_center = [DEFAULT_LATITUDE, DEFAULT_LONGITUDE]

        site_map = folium.Map(location=map_center, zoom_start=6, tiles='CartoDB positron')

        # Draw Threats
        if threats:
            for threat in threats:
                # Add a heatmap style circle for risk intensity
                folium.Circle(
                    location=[threat.lat, threat.lon],
                    radius=threat.radius_km * 1000, # Meters
                    color='red',
                    weight=1,
                    fill=True,
                    fill_opacity=0.2,
                    popup=f"THREAT: {threat.name}"
                ).add_to(site_map)

                # Inner core
                folium.Circle(
                    location=[threat.lat, threat.lon],
                    radius=threat.radius_km * 1000 * 0.5,
                    color='darkred',
                    weight=0,
                    fill=True,
                    fill_opacity=0.4,
                ).add_to(site_map)

        # Colors for sorties
        colors = ['blue', 'green', 'purple', 'orange', 'darkred', 'cadetblue']

        # Add Base Marker
        folium.Marker(
            location=[float(start_site['Latitude']), float(start_site['Longitude'])],
            popup=folium.Popup(f"<b>BASE: {start_site['Site Name']}</b>", max_width=300),
            icon=folium.Icon(color='red', icon='home', prefix='fa')
        ).add_to(site_map)

        for i, sortie in enumerate(sorties):
            color = colors[i % len(colors)]
            if sortie['status'] == 'RED':
                color = 'gray' # Dim the invalid routes

            # Draw Detailed Path if available, else fallback to site points
            detailed_path = sortie.get('detailed_path', [])
            if detailed_path and len(detailed_path) > 1:
                folium.PolyLine(
                    locations=detailed_path,
                    color=color,
                    weight=4,
                    opacity=0.8,
                    tooltip=f"Sortie {sortie['id']} Path ({sortie['status']})"
                ).add_to(site_map)
            else:
                # Fallback
                points = []
                for site in sortie['route']:
                     points.append([float(site['Latitude']), float(site['Longitude'])])
                if len(points) > 1:
                     folium.PolyLine(locations=points, color=color, weight=4).add_to(site_map)

            # Add markers for sites
            for j, site in enumerate(sortie['route']):
                try:
                    lat = float(site['Latitude'])
                    lon = float(site['Longitude'])

                    if site['Site Name'] != start_site['Site Name']:
                        folium.Marker(
                            location=[lat, lon],
                            popup=folium.Popup(f"<b>S{sortie['id']}-#{j+1}: {site['Site Name']}</b>", max_width=300),
                            icon=folium.Icon(color=color, icon='info-sign', prefix='fa')
                        ).add_to(site_map)
                except:
                    continue

        # Add Simulation Markers
        if simulation_states:
            for state in simulation_states:
                s_id = state['id']
                # find color
                # sorties list is 0-indexed, s_id is 1-indexed usually in my logic
                idx = (s_id - 1) % len(colors)
                color = colors[idx]

                folium.Marker(
                    location=[state['lat'], state['lon']],
                    icon=folium.Icon(color=color, icon='plane', prefix='fa'),
                    popup=f"Sortie #{s_id} Current Pos"
                ).add_to(site_map)

        return site_map._repr_html_()

    # --- Simulation Logic ---
    def simulate_mission(progress, sorties, all_sites_df, threats):
        if not sorties:
            return gr.update(), {"status": "No Active Mission"}

        if not MissionSimulator:
             return gr.update(), {"error": "Simulator not loaded"}

        # Initialize simulator just-in-time (cheap)
        sim = MissionSimulator(sorties)
        states = sim.get_state_at_progress(progress)

        # Telemetry
        telemetry = {}
        for state in states:
            s_id = state['id']
            telemetry[f"Sortie_{s_id}"] = {
                "lat": round(state['lat'], 4),
                "lon": round(state['lon'], 4),
                "status": "In Transit"
            }

        # We need to regenerate the map with the new markers
        # We need the start site for the map gen. We can find it from the first sortie.
        start_site = sorties[0]['route'][0] # Approximation

        map_html = generate_multi_sortie_map(all_sites_df, sorties, start_site, threats, simulation_states=states)

        return map_html, telemetry

    # Connect Mission Planner Events
    mp_execute_btn.click(
        fn=generate_mission_plan,
        inputs=[
            mp_site_selector,
            mp_start_selector,
            fleet_uav_count,
            fleet_suv_count,
            fleet_helo_count,
            fleet_sedan_count,
            all_sites_df_state,
            threats_state
        ],
        outputs=[mp_briefing_output, mp_map_output, mp_download_file, mission_sorties_state]
    )

    # --- Wargame Handlers ---
    def generate_coas_handler(selected_sites, start_site_name, n_uav, n_suv, n_helo, n_sedan, all_sites_df, threats):
        # We need to reuse logic from generate_mission_plan but for 3 strategies.
        # This is duplication but for the sake of clarity and separation in this "transformative" task, we do it.
        # Ideally, refactor fleet construction.

        # Fleet Construction
        fleet = []
        try:
            for _ in range(int(n_uav)): fleet.append(ASSET_REGISTRY.get("Drone (UAV)"))
            for _ in range(int(n_suv)): fleet.append(ASSET_REGISTRY.get("Ground Team (SUV)"))
            for _ in range(int(n_helo)): fleet.append(ASSET_REGISTRY.get("Air Support (Helo)"))
            for _ in range(int(n_sedan)): fleet.append(ASSET_REGISTRY.get("Covert Ops (Sedan)"))
        except Exception:
            pass
        fleet = [f for f in fleet if f is not None]

        if not fleet or not selected_sites:
             return {}, gr.update(), gr.update(), gr.update() # Fail silently or handle error

        # Prepare sites
        selected_df = all_sites_df[all_sites_df['Site Name'].isin(selected_sites)]
        if start_site_name and start_site_name in selected_sites:
            start_row = selected_df[selected_df['Site Name'] == start_site_name].iloc[0]
        else:
            start_row = selected_df.iloc[0]
        start_site_dict = start_row.to_dict()
        target_sites = [r.to_dict() for _, r in selected_df.iterrows() if r['Site Name'] != start_site_dict['Site Name']]

        planner = StrategicSortiePlanner(start_site_dict, target_sites)
        coas = planner.generate_coas(fleet, threats)

        return coas, gr.update(value="Ready for Sim"), gr.update(value="Ready for Sim"), gr.update(value="Ready for Sim")

    def run_monte_carlo(coas, threats):
        if not coas:
            return {}, {}, {}, None

        engine = MonteCarloEngine(num_runs=50) # Keep it fast for demo

        stats_speed = engine.simulate_plan(coas['SPEED'], threats)
        stats_stealth = engine.simulate_plan(coas['STEALTH'], threats)
        stats_efficiency = engine.simulate_plan(coas['EFFICIENCY'], threats)

        # Create Plot
        import matplotlib.pyplot as plt

        strategies = ['Speed', 'Stealth', 'Efficiency']
        success_rates = [stats_speed.success_rate, stats_stealth.success_rate, stats_efficiency.success_rate]
        times = [stats_speed.avg_time, stats_stealth.avg_time, stats_efficiency.avg_time]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        ax1.bar(strategies, success_rates, color=['blue', 'green', 'orange'])
        ax1.set_title("Mission Success Probability (%)")
        ax1.set_ylim(0, 100)

        ax2.bar(strategies, times, color=['blue', 'green', 'orange'])
        ax2.set_title("Avg Mission Duration (Hrs)")

        plt.tight_layout()

        return (
            stats_speed.__dict__,
            stats_stealth.__dict__,
            stats_efficiency.__dict__,
            fig
        )

    def select_coa(coas, strategy_key, all_sites_df, threats):
        sorties = coas.get(strategy_key, [])
        # We need to update the main map and briefing with this choice.
        # Reuse generate_mission_plan return format

        # Start site is in the sorties
        if not sorties:
            return "No Plan", "<p>Empty</p>", None, []

        start_site_name = sorties[0]['route'][0]['Site Name']
        start_row = all_sites_df[all_sites_df['Site Name'] == start_site_name].iloc[0]
        start_site_dict = start_row.to_dict()

        map_html = generate_multi_sortie_map(all_sites_df, sorties, start_site_dict, threats)

        briefing = f"# 📜 Strategic Operational Briefing ({strategy_key})\n\n"
        briefing += "** SELECTED FROM WARGAME ANALYSIS **\n\n"
        # ... (Simplified briefing regeneration or reuse logic) ...
        # For brevity, let's just output basic info
        briefing += f"Plan Strategy: {strategy_key}\n"
        briefing += f"Sorties: {len(sorties)}\n"

        return briefing, map_html, None, sorties

    wr_generate_btn.click(
        fn=generate_coas_handler,
        inputs=[mp_site_selector, mp_start_selector, fleet_uav_count, fleet_suv_count, fleet_helo_count, fleet_sedan_count, all_sites_df_state, threats_state],
        outputs=[wargame_results_state, wr_stats_speed, wr_stats_stealth, wr_stats_efficiency]
    )

    wr_simulate_btn.click(
        fn=run_monte_carlo,
        inputs=[wargame_results_state, threats_state],
        outputs=[wr_stats_speed, wr_stats_stealth, wr_stats_efficiency, wr_plot_output]
    )

    # Selecting a COA updates the Mission Planner Tab outputs
    wr_select_speed_btn.click(
        fn=partial(select_coa, strategy_key='SPEED'),
        inputs=[wargame_results_state, all_sites_df_state, threats_state],
        outputs=[mp_briefing_output, mp_map_output, mp_download_file, mission_sorties_state]
    )

    wr_select_stealth_btn.click(
        fn=partial(select_coa, strategy_key='STEALTH'),
        inputs=[wargame_results_state, all_sites_df_state, threats_state],
        outputs=[mp_briefing_output, mp_map_output, mp_download_file, mission_sorties_state]
    )

    wr_select_efficiency_btn.click(
        fn=partial(select_coa, strategy_key='EFFICIENCY'),
        inputs=[wargame_results_state, all_sites_df_state, threats_state],
        outputs=[mp_briefing_output, mp_map_output, mp_download_file, mission_sorties_state]
    )

    mp_sim_slider.change(
        fn=simulate_mission,
        inputs=[mp_sim_slider, mission_sorties_state, all_sites_df_state, threats_state],
        outputs=[mp_map_output, mp_telemetry_output]
    )

    # --- Event Handlers (search_box.submit) ---
    def update_search_results(query, current_all_sites_df):
        if not query:
            filtered_df = current_all_sites_df.copy() # Use copy to avoid modifying state directly
        else:
            q_lower = query.lower()
            df_searchable = current_all_sites_df[
                current_all_sites_df['Site Name'].astype(str).str.lower().str.contains(q_lower) |
                current_all_sites_df['Description'].astype(str).str.lower().str.contains(q_lower)
            ]
            filtered_df = df_searchable

        new_map_html = generate_map_html(filtered_df)

        return new_map_html, filtered_df

    def back_to_list_view_fn():
        return gr.update(visible=True), gr.update(visible=False)

    # --- Function to update data based on country selection ---
    def update_country_data(selected_country):
        print(f"Country selected: {selected_country}. Loading data...")
        new_df = load_data(selected_country)

        if new_df.empty:
            updated_map_html = "<p style='text-align:center; color:grey;'>Map data not available: Data for selected country not found.</p>"
            empty_df_for_state = pd.DataFrame(columns=EXPECTED_COLUMNS)
            updated_title = f"# {selected_country} UNESCO World Heritage Sites Dashboard - Data not found"
            return [
                empty_df_for_state,
                empty_df_for_state.copy(),
                updated_map_html,
                gr.update(value=""),
                gr.update(value=updated_title),
                gr.update(visible=True),
                gr.update(visible=False, value="")
            ]
        else:
            updated_map_html = generate_map_html(new_df)
            updated_title = f"# {selected_country} UNESCO World Heritage Sites Dashboard"
            return [
                new_df,
                new_df.copy(),
                updated_map_html,
                gr.update(value=""),
                gr.update(value=updated_title),
                gr.update(visible=False),
                gr.update(visible=False, value="")
            ]

    # --- Function to show scraper instructions ---
    def show_scraper_instructions(country_name):
        instructions = f"""To generate data for **{country_name}**, please run the following command in your terminal from the project's root directory:

```bash
python scraper.py --country "{country_name}"
```

After running the command, please re-select **{country_name}** from the dropdown in the app to load the new data."""
        return gr.update(value=instructions, visible=True)

    # --- Connect Event Handlers ---
    search_box.submit(
        fn=update_search_results,
        inputs=[
            search_box, 
            all_sites_df_state, # all_sites_df_state will hold current country's data
        ],
        outputs=[map_output, filtered_sites_df_state]
    )

    back_to_list_btn.click(
        fn=back_to_list_view_fn,
        inputs=[],
        outputs=[main_content_area, detailed_view_area]
    )

    # Function that wraps update_country_data but also updates mission planner choices
    def update_country_and_mp_choices(selected_country):
        # 1. Update general data
        result_list = update_country_data(selected_country)
        # result_list order:
        # [0] all_sites_df_state (new_df)
        # [1] filtered_sites_df_state (new_df copy)
        # [2] map_output
        # ... others ...

        new_df = result_list[0]

        # 2. Update Mission Planner Choices
        mp_choices, mp_start = update_mp_choices(new_df)

        # Combine results
        return result_list + [mp_choices, mp_start]

    country_dropdown.change(
        fn=update_country_and_mp_choices,
        inputs=[
            country_dropdown, # selected_country
        ],
        outputs=[
            all_sites_df_state,       # Update the main DataFrame state
            filtered_sites_df_state,  # Update the filtered DataFrame state (reset to all for new country)
            map_output,               # Update the map
            search_box,               # Clear search box
            main_title_md,            # Update the main title
            scrape_data_button,       # Show/hide scrape button
            scraper_instructions_md,  # Show/hide instructions
            mp_site_selector,         # Update MP site selector choices
            mp_start_selector         # Update MP start selector choices
        ]
    )

    scrape_data_button.click(
        fn=show_scraper_instructions,
        inputs=[country_dropdown], # Pass the selected country name
        outputs=[scraper_instructions_md]
    )

    # --- App Load Action ---
    def initial_load(current_all_sites_df):
        # current_all_sites_df is initial_df (Italy data) on first load
        # No filtering on initial load, so filtered_df is a copy of current_all_sites_df
        filtered_df_on_load = current_all_sites_df.copy()

        initial_map_html = generate_map_html(filtered_df_on_load)

        # Also initialize MP choices
        mp_choices, mp_start = update_mp_choices(current_all_sites_df)

        return initial_map_html, filtered_df_on_load, mp_choices, mp_start

    app.load(
        fn=initial_load,
        inputs=[
            all_sites_df_state, # This will pass the initial_df (Italy data)
        ],
        outputs=[map_output, filtered_sites_df_state, mp_site_selector, mp_start_selector]
    )

if __name__ == "__main__":
    from threading import Thread

    def run_flask_app():
        app_flask.run(port=5000)

    flask_thread = Thread(target=run_flask_app)
    flask_thread.daemon = True
    flask_thread.start()
    # Initial check refers to Italy data by default due to load_data() default
    italy_data_file = os.path.join("data", "unesco_sites_italy.csv")
    if initial_df.empty and not os.path.exists(italy_data_file):
        print(f"CRITICAL: Default data file for Italy ('{italy_data_file}') not found. The app will launch but may be mostly empty.")
        print("Please ensure 'scraper.py' has been run successfully for 'Italy'.")
    elif initial_df.empty:
        print(f"WARNING: Default data file for Italy ('{italy_data_file}') was found but appears to be empty or failed to load correctly.")

    # Check for France data file as well, just as a common example.
    france_data_file = os.path.join("data", "unesco_sites_france.csv")
    if not os.path.exists(france_data_file):
        print(f"INFO: Data file for France ('{france_data_file}') not found. You may need to run the scraper for France.")

    print("Launching Gradio app...")
    app.launch()
