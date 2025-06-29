import unittest
import subprocess
import os
import sys
import pandas as pd

class TestScraperExecution(unittest.TestCase):

    def test_scrape_france(self):
        """
        Tests running scraper.py with --country "France" and verifies CSV output.
        """
        country_name = "France"
        country_name_file_part = country_name.lower().replace(' ', '_')
        output_csv_filename = f"unesco_sites_{country_name_file_part}.csv"
        output_csv_path = os.path.join("data", output_csv_filename)

        # Ensure the data directory exists (scraper should create it, but good for cleanup)
        os.makedirs("data", exist_ok=True)

        # Remove the CSV file if it exists from a previous run
        if os.path.exists(output_csv_path):
            os.remove(output_csv_path)
            print(f"Removed existing file: {output_csv_path}")

        # Command to run the scraper script
        # We need to ensure we're using the same Python interpreter that's running the test
        command = [sys.executable, "scraper.py", "--country", country_name]

        print(f"Running command: {' '.join(command)}")

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False, # Don't raise exception on non-zero exit, we'll check manually
                timeout=180 # Add a timeout (e.g., 180 seconds)
            )
        except subprocess.TimeoutExpired:
            self.fail(f"Scraper script timed out after 180 seconds.")
            return

        print(f"Scraper stdout:\n{result.stdout}")
        if result.stderr:
            print(f"Scraper stderr:\n{result.stderr}")

        # 1. Check if the script ran successfully (exit code 0)
        self.assertEqual(result.returncode, 0, f"Scraper script exited with code {result.returncode}. Stderr: {result.stderr}")

        # 2. Check if the CSV file was created
        self.assertTrue(os.path.exists(output_csv_path), f"Output CSV file was not created: {output_csv_path}")

        # 3. Check if the CSV file is not empty
        try:
            file_size = os.path.getsize(output_csv_path)
            self.assertGreater(file_size, 0, f"Output CSV file is empty: {output_csv_path}")
            print(f"Output CSV file {output_csv_path} created with size: {file_size} bytes.")
        except OSError as e:
            self.fail(f"Could not get size of output CSV file: {output_csv_path}. Error: {e}")

        # 4. Optional: Validate CSV content (e.g., can it be read by pandas? does it have rows?)
        try:
            df = pd.read_csv(output_csv_path)
            self.assertGreater(len(df), 0, f"CSV file {output_csv_path} was created but contains no data rows.")
            print(f"CSV file {output_csv_path} successfully read by pandas, found {len(df)} rows.")
        except pd.errors.EmptyDataError:
            self.fail(f"CSV file {output_csv_path} is empty or malformed (pandas EmptyDataError).")
        except Exception as e:
            self.fail(f"Failed to read or validate CSV file {output_csv_path} with pandas. Error: {e}")


if __name__ == '__main__':
    unittest.main()
