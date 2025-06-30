import unittest
import pandas as pd
import os
import sys
import tempfile
import shutil

# Add the parent directory to the path to import app modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestAppFunctionality(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.test_data_dir = tempfile.mkdtemp()
        self.original_data_dir = None
        
    def tearDown(self):
        """Clean up after each test."""
        if os.path.exists(self.test_data_dir):
            shutil.rmtree(self.test_data_dir)
            
    def test_load_data_with_valid_csv(self):
        """Test load_data function with a valid CSV file."""
        from app import load_data, EXPECTED_COLUMNS
        
        # Create a test CSV file with all expected columns
        test_csv_path = os.path.join(self.test_data_dir, "unesco_sites_test.csv")
        test_data = {
            'Site Name': ['Test Site 1', 'Test Site 2'],
            'Image URL': ['http://example.com/img1.jpg', 'http://example.com/img2.jpg'],
            'Location': ['Test Location 1', 'Test Location 2'],
            'Year Listed': ['2000', '2001'],
            'UNESCO Data': ['Test Data 1', 'Test Data 2'],
            'Description': ['Test Description 1', 'Test Description 2'],
            'Latitude': [45.0, 46.0],
            'Longitude': [9.0, 10.0]
        }
        
        df = pd.DataFrame(test_data)
        df.to_csv(test_csv_path, index=False)
        
        # Temporarily change the data directory
        original_cwd = os.getcwd()
        os.chdir(os.path.dirname(test_csv_path))
        
        try:
            # Mock the file path by creating the expected directory structure
            os.makedirs("data", exist_ok=True)
            shutil.copy(test_csv_path, "data/unesco_sites_test.csv")
            
            # Test loading the data
            result_df = load_data("test")
            
            # Verify the result
            self.assertFalse(result_df.empty, "DataFrame should not be empty")
            self.assertEqual(len(result_df), 2, "Should load 2 rows")
            self.assertEqual(list(result_df.columns), EXPECTED_COLUMNS, "Should have all expected columns")
            
        finally:
            os.chdir(original_cwd)
            
    def test_load_data_missing_coordinates(self):
        """Test load_data function when CSV is missing coordinate columns."""
        from app import load_data, EXPECTED_COLUMNS, DEFAULT_LATITUDE, DEFAULT_LONGITUDE
        
        # Create a test CSV file without coordinate columns
        test_csv_path = os.path.join(self.test_data_dir, "unesco_sites_test_no_coords.csv")
        test_data = {
            'Site Name': ['Test Site 1'],
            'Image URL': ['http://example.com/img1.jpg'],
            'Location': ['Test Location 1'],
            'Year Listed': ['2000'],
            'UNESCO Data': ['Test Data 1'],
            'Description': ['Test Description 1']
        }
        
        df = pd.DataFrame(test_data)
        df.to_csv(test_csv_path, index=False)
        
        # Temporarily change the data directory
        original_cwd = os.getcwd()
        os.chdir(os.path.dirname(test_csv_path))
        
        try:
            # Mock the file path by creating the expected directory structure
            os.makedirs("data", exist_ok=True)
            shutil.copy(test_csv_path, "data/unesco_sites_test_no_coords.csv")
            
            # Test loading the data
            result_df = load_data("test_no_coords")
            
            # Verify the result
            self.assertFalse(result_df.empty, "DataFrame should not be empty")
            self.assertTrue('Latitude' in result_df.columns, "Should have Latitude column")
            self.assertTrue('Longitude' in result_df.columns, "Should have Longitude column")
            
            # Check default coordinates are used
            self.assertEqual(result_df['Latitude'].iloc[0], DEFAULT_LATITUDE, "Should use default latitude")
            self.assertEqual(result_df['Longitude'].iloc[0], DEFAULT_LONGITUDE, "Should use default longitude")
            
        finally:
            os.chdir(original_cwd)
            
    def test_generate_map_html_with_valid_data(self):
        """Test map generation with valid coordinate data."""
        from app import generate_map_html
        
        # Create test data with valid coordinates
        test_data = pd.DataFrame({
            'Site Name': ['Test Site 1', 'Test Site 2'],
            'Latitude': [45.0, 46.0],
            'Longitude': [9.0, 10.0],
            'Image URL': ['http://example.com/img1.jpg', 'N/A'],
            'Location': ['Milan', 'Turin'],
            'Year Listed': ['2000', '2001']
        })
        
        result = generate_map_html(test_data)
        
        # Verify the result
        self.assertIsInstance(result, str, "Should return HTML string")
        self.assertNotIn("Map data is unavailable", result, "Should not show unavailable message")
        
    def test_generate_map_html_with_empty_data(self):
        """Test map generation with empty data."""
        from app import generate_map_html
        
        # Create empty DataFrame
        empty_df = pd.DataFrame()
        
        result = generate_map_html(empty_df)
        
        # Verify the result
        self.assertIn("Map data is unavailable", result, "Should show unavailable message")
        
    def test_show_site_details_with_missing_site(self):
        """Test show_site_details function with non-existent site."""
        from app import show_site_details
        
        # Create test data
        test_data = pd.DataFrame({
            'Site Name': ['Test Site 1'],
            'Image URL': ['http://example.com/img1.jpg'],
            'Description': ['Test Description'],
            'Location': ['Test Location'],
            'Year Listed': ['2000'],
            'UNESCO Data': ['Test Data']
        })
        
        # Test with non-existent site
        result = show_site_details("Non-existent Site", test_data)
        
        # Verify the result
        self.assertIsInstance(result, dict, "Should return dictionary")
        # The function should handle the missing site gracefully
        
    def test_coordinate_validation_in_scraper_output(self):
        """Test that scraper output includes coordinate validation."""
        # This test would verify that any CSV generated by the scraper
        # includes Latitude and Longitude columns
        
        expected_columns = ['Site Name', 'Image URL', 'Location', 'Year Listed', 
                          'UNESCO Data', 'Description', 'Latitude', 'Longitude']
        
        # Check if Italy data exists and has correct columns
        italy_csv_path = "data/unesco_sites_italy.csv"
        if os.path.exists(italy_csv_path):
            try:
                df = pd.read_csv(italy_csv_path)
                
                # Check if coordinate columns exist
                missing_coords = []
                if 'Latitude' not in df.columns:
                    missing_coords.append('Latitude')
                if 'Longitude' not in df.columns:
                    missing_coords.append('Longitude')
                    
                if missing_coords:
                    self.fail(f"Missing coordinate columns in existing data: {missing_coords}. "
                            f"Available columns: {list(df.columns)}")
                    
                # Verify coordinate data types
                if 'Latitude' in df.columns:
                    lat_numeric = pd.to_numeric(df['Latitude'], errors='coerce')
                    self.assertFalse(lat_numeric.isna().all(), "Latitude values should be numeric")
                    
                if 'Longitude' in df.columns:
                    lon_numeric = pd.to_numeric(df['Longitude'], errors='coerce')
                    self.assertFalse(lon_numeric.isna().all(), "Longitude values should be numeric")
                    
            except Exception as e:
                self.fail(f"Error reading Italy CSV file: {e}")
        else:
            self.skipTest("Italy CSV file not found - run scraper first")


if __name__ == '__main__':
    unittest.main()