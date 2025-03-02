import requests
import json
import os

def download_bestiary_data():
    # Define URLs for GitHub raw content
    INDEX_URL = "https://raw.githubusercontent.com/5etools-mirror-3/5etools-src/main/data/bestiary/index.json"
    BASE_URL = "https://raw.githubusercontent.com/5etools-mirror-3/5etools-src/main/data/bestiary/"
    
    # Create directory for downloaded files
    os.makedirs("bestiary_files", exist_ok=True)
    
    # Check if index file exists locally
    if os.path.exists("bestiary_files/index.json"):
        print("Loading bestiary index from local file...")
        with open("bestiary_files/index.json", "r", encoding="utf-8") as f:
            index_data = json.load(f)
        print(f"Index loaded successfully. Found {len(index_data)} entries.")
    else:
        print("Downloading bestiary index...")
        try:
            # Get the index file
            response = requests.get(INDEX_URL)
            response.raise_for_status()
            index_data = response.json()
            
            # Save the index file
            with open("bestiary_files/index.json", "w", encoding="utf-8") as f:
                json.dump(index_data, f, indent=2)
                
            print(f"Index downloaded successfully. Found {len(index_data)} entries.")
        except Exception as e:
            print(f"Error downloading index: {e}")
            return None
    
    # Filter for only bestiary files
    bestiary_files = {key: filename for key, filename in index_data.items() 
                     if filename.startswith("bestiary-")}
    
    print(f"Found {len(bestiary_files)} bestiary files to process.")
    
    # Download each bestiary file (if it doesn't exist locally)
    all_monsters = []
    failed_files = []
    
    for key, filename in bestiary_files.items():
        local_path = f"bestiary_files/{filename}"
        
        # Check if file already exists locally
        if os.path.exists(local_path):
            print(f"Loading {filename} from local file...")
            try:
                with open(local_path, "r", encoding="utf-8") as f:
                    file_data = json.load(f)
            except Exception as e:
                print(f"  Error loading local file {filename}: {e}")
                failed_files.append(filename)
                continue
        else:
            file_url = BASE_URL + filename
            print(f"Downloading {filename}...")
            
            try:
                response = requests.get(file_url)
                response.raise_for_status()
                file_data = response.json()
                
                # Save the file locally
                with open(local_path, "w", encoding="utf-8") as f:
                    json.dump(file_data, f, indent=2)
            except Exception as e:
                print(f"  Error downloading {filename}: {e}")
                failed_files.append(filename)
                continue
        
        # Extract monster data if present
        if "monster" in file_data:
            monsters_in_file = file_data["monster"]
            all_monsters.extend(monsters_in_file)
            print(f"  Added {len(monsters_in_file)} monsters from {filename}")
        else:
            print(f"  No monsters found in {filename}")
    
    # Create combined monster dictionary
    combined_data = {"monster": all_monsters}
    
    # Save combined data
    with open("all_monsters.json", "w", encoding="utf-8") as f:
        json.dump(combined_data, f, indent=2)
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed {len(bestiary_files) - len(failed_files)} files")
    print(f"Combined {len(all_monsters)} monsters into 'all_monsters.json'")
    
    if failed_files:
        print(f"Failed to process {len(failed_files)} files: {', '.join(failed_files)}")
    
    return combined_data

def main():
    # Download and combine all bestiary data
    combined_data = download_bestiary_data()
    
    if combined_data:
        # Extract just the monster list
        monsters = combined_data["monster"]
        
        print("\nAccessing the monster data in Python:")
        print(f"Total monsters collected: {len(monsters)}")
        
        # Show sample of the data
        if monsters:
            first_monster = monsters[0]
            print(f"\nSample monster: {first_monster.get('name', 'Unknown')}")
            print(f"Type: {first_monster.get('type', 'Unknown')}")
            print(f"Challenge Rating: {first_monster.get('cr', 'Unknown')}")
        
        print("\nYou can now work with the 'monsters' list in your Python code")
        return monsters
    else:
        print("Failed to download and process bestiary data")
        return None

if __name__ == "__main__":
    monsters = main()