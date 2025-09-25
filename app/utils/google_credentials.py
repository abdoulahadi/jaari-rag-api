"""
Google Cloud credentials management for deployment environments
"""
import os
import json
import tempfile
from pathlib import Path
from typing import Optional


def setup_google_credentials():
    """
    Setup Google Cloud credentials for translation service.
    
    Returns:
        str: Path to the credentials file, or None if not available
    """
    # First, try to get credentials from JSON environment variable (for Render/production)
    credentials_json = os.getenv('GOOGLE_APPLICATION_CREDENTIALS_JSON')
    
    if credentials_json:
        try:
            # Decode if it's base64 encoded
            import base64
            try:
                credentials_json = base64.b64decode(credentials_json).decode('utf-8')
            except:
                # If decoding fails, assume it's already plain JSON
                pass
            
            # Parse JSON to validate
            credentials_data = json.loads(credentials_json)
            
            # Create a temporary file with the credentials
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
            json.dump(credentials_data, temp_file, indent=2)
            temp_file.close()
            
            # Set the environment variable to point to the temp file
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = temp_file.name
            
            print(f"Google Cloud credentials loaded from environment variable")
            return temp_file.name
            
        except json.JSONDecodeError as e:
            print(f"Error parsing Google Cloud credentials JSON: {e}")
            return None
        except Exception as e:
            print(f"Error setting up Google Cloud credentials: {e}")
            return None
    
    # Fallback: try to use existing file-based credentials (for local development)
    credentials_file = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    
    if credentials_file and Path(credentials_file).exists():
        print(f"Using existing Google Cloud credentials file: {credentials_file}")
        return credentials_file
    
    # Look for default locations
    default_locations = [
        './translate-jaari-065fa764be8a.json',
        '../translate-jaari-065fa764be8a.json',
        '/opt/render/project/src/translate-jaari-065fa764be8a.json'
    ]
    
    for location in default_locations:
        if Path(location).exists():
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = location
            print(f"Found Google Cloud credentials at: {location}")
            return location
    
    print("Warning: Google Cloud credentials not found. Translation features may not work.")
    return None


def get_google_credentials_path():
    """
    Get the path to Google Cloud credentials file.
    Sets up credentials if needed.
    
    Returns:
        str: Path to credentials file or None if not available
    """
    return setup_google_credentials()
