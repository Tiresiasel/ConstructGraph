#!/usr/bin/env python3
"""
View JSON Outputs from Build Graph Process

This script helps you easily view the JSON outputs extracted by the LLM
without having to dig through long log files.
"""

import re
import json
from pathlib import Path
from datetime import datetime

def extract_json_from_logs(log_file_path):
    """Extract JSON outputs from the log file."""
    json_outputs = []
    
    if not log_file_path.exists():
        print(f"Log file not found: {log_file_path}")
        return json_outputs
    
    with open(log_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by the separator lines
    sections = content.split('=' * 80)
    
    for section in sections:
        if 'COMPLETE JSON OUTPUT FOR:' in section:
            # Extract filename and timestamp
            lines = section.strip().split('\n')
            filename = None
            timestamp = None
            json_content = []
            
            for line in lines:
                if 'COMPLETE JSON OUTPUT FOR:' in line:
                    filename = line.split('COMPLETE JSON OUTPUT FOR:')[1].strip()
                elif 'Timestamp:' in line:
                    timestamp = line.split('Timestamp:')[1].strip()
                elif line.strip() and not line.startswith('|'):
                    json_content.append(line)
            
            if filename and json_content:
                # Try to parse the JSON content
                try:
                    # Join the JSON content and parse
                    json_str = '\n'.join(json_content)
                    # Find the JSON part (between "Raw JSON Response:" and the end)
                    if 'Raw JSON Response:' in json_str:
                        json_str = json_str.split('Raw JSON Response:')[1].strip()
                    
                    parsed_json = json.loads(json_str)
                    json_outputs.append({
                        'filename': filename,
                        'timestamp': timestamp,
                        'json_data': parsed_json
                    })
                except json.JSONDecodeError as e:
                    print(f"Warning: Could not parse JSON for {filename}: {e}")
                    # Store the raw content for debugging
                    json_outputs.append({
                        'filename': filename,
                        'timestamp': timestamp,
                        'json_data': None,
                        'raw_content': json_str
                    })
    
    return json_outputs

def display_json_summary(json_outputs):
    """Display a summary of all JSON outputs."""
    if not json_outputs:
        print("No JSON outputs found in logs.")
        return
    
    print(f"\n{'='*80}")
    print(f"JSON OUTPUTS SUMMARY")
    print(f"Total files processed: {len(json_outputs)}")
    print(f"{'='*80}")
    
    for i, output in enumerate(json_outputs, 1):
        filename = output['filename']
        timestamp = output['timestamp']
        json_data = output['json_data']
        
        print(f"\n{i}. {filename}")
        print(f"   Timestamp: {timestamp}")
        
        if json_data:
            # Count different types of extracted data
            constructs = len(json_data.get('constructs', []))
            relationships = len(json_data.get('relationships', []))
            measurements = len(json_data.get('measurements', []))
            theories = len(json_data.get('core_theories', []))
            
            print(f"   Extracted:")
            print(f"     - Constructs: {constructs}")
            print(f"     - Relationships: {relationships}")
            print(f"     - Measurements: {measurements}")
            print(f"     - Theories: {theories}")
            
            # Show paper metadata if available
            if 'paper_metadata' in json_data:
                metadata = json_data['paper_metadata']
                title = metadata.get('title', 'N/A')
                authors = metadata.get('authors', 'N/A')
                year = metadata.get('publication_year', 'N/A')
                print(f"   Paper: {title}")
                print(f"   Authors: {authors}")
                print(f"   Year: {year}")
        else:
            print(f"   Status: JSON parsing failed")
    
    print(f"\n{'='*80}")

def view_specific_json(json_outputs, filename=None, index=None):
    """View the complete JSON for a specific file."""
    if not json_outputs:
        print("No JSON outputs available.")
        return
    
    if filename:
        # Find by filename
        target_output = None
        for output in json_outputs:
            if filename in output['filename']:
                target_output = output
                break
        
        if not target_output:
            print(f"File '{filename}' not found in JSON outputs.")
            return
    elif index is not None:
        # Find by index
        if 1 <= index <= len(json_outputs):
            target_output = json_outputs[index - 1]
        else:
            print(f"Index {index} out of range. Available: 1-{len(json_outputs)}")
            return
    else:
        # Show selection menu
        print("\nSelect a file to view complete JSON:")
        for i, output in enumerate(json_outputs, 1):
            print(f"{i}. {output['filename']}")
        
        try:
            choice = input(f"\nEnter number (1-{len(json_outputs)}): ").strip()
            index = int(choice)
            if 1 <= index <= len(json_outputs):
                target_output = json_outputs[index - 1]
            else:
                print("Invalid selection.")
                return
        except (ValueError, KeyboardInterrupt):
            print("Invalid input or cancelled.")
            return
    
    # Display the complete JSON
    print(f"\n{'='*80}")
    print(f"COMPLETE JSON FOR: {target_output['filename']}")
    print(f"Timestamp: {target_output['timestamp']}")
    print(f"{'='*80}")
    
    if target_output['json_data']:
        print(json.dumps(target_output['json_data'], indent=2, ensure_ascii=False))
    else:
        print("JSON parsing failed. Raw content:")
        print(target_output.get('raw_content', 'No raw content available'))
    
    print(f"{'='*80}")

def search_in_json(json_outputs, search_term):
    """Search for specific terms in the JSON outputs."""
    if not json_outputs:
        print("No JSON outputs available for search.")
        return
    
    print(f"\nSearching for: '{search_term}'")
    print(f"{'='*80}")
    
    found_in = []
    
    for output in json_outputs:
        if not output['json_data']:
            continue
            
        filename = output['filename']
        json_data = output['json_data']
        
        # Search in constructs
        for construct in json_data.get('constructs', []):
            term = construct.get('term', '').lower()
            definition = construct.get('definition', '').lower()
            if search_term.lower() in term or search_term.lower() in definition:
                found_in.append({
                    'filename': filename,
                    'type': 'construct',
                    'term': construct.get('term'),
                    'definition': construct.get('definition', '')[:100] + '...' if len(construct.get('definition', '')) > 100 else construct.get('definition', '')
                })
        
        # Search in relationships
        for rel in json_data.get('relationships', []):
            subject = rel.get('subject_term', '').lower()
            object_term = rel.get('object_term', '').lower()
            if search_term.lower() in subject or search_term.lower() in object_term:
                found_in.append({
                    'filename': filename,
                    'type': 'relationship',
                    'subject': rel.get('subject_term'),
                    'object': rel.get('object_term'),
                    'status': rel.get('status')
                })
    
    if found_in:
        print(f"Found {len(found_in)} matches:")
        for i, match in enumerate(found_in, 1):
            print(f"\n{i}. File: {match['filename']}")
            if match['type'] == 'construct':
                print(f"   Type: Construct")
                print(f"   Term: {match['term']}")
                print(f"   Definition: {match['definition']}")
            else:
                print(f"   Type: Relationship")
                print(f"   {match['subject']} -> {match['object']}")
                print(f"   Status: {match['status']}")
    else:
        print(f"No matches found for '{search_term}'")
    
    print(f"{'='*80}")

def main():
    """Main function with interactive menu."""
    logs_dir = Path(__file__).parent / "logs"
    json_log_file = logs_dir / "json_outputs.log"
    
    print("JSON Output Viewer for Build Graph Process")
    print("=" * 50)
    
    # Extract JSON outputs from logs
    json_outputs = extract_json_from_logs(json_log_file)
    
    if not json_outputs:
        print("No JSON outputs found. Make sure to run build_graph.py first.")
        return
    
    while True:
        print(f"\n{'='*50}")
        print("Available actions:")
        print("1. View summary of all JSON outputs")
        print("2. View complete JSON for a specific file")
        print("3. Search for specific terms in JSON outputs")
        print("4. Exit")
        print(f"{'='*50}")
        
        try:
            choice = input("Enter your choice (1-4): ").strip()
            
            if choice == '1':
                display_json_summary(json_outputs)
            
            elif choice == '2':
                view_specific_json(json_outputs)
            
            elif choice == '3':
                search_term = input("Enter search term: ").strip()
                if search_term:
                    search_in_json(json_outputs, search_term)
                else:
                    print("Search term cannot be empty.")
            
            elif choice == '4':
                print("Goodbye!")
                break
            
            else:
                print("Invalid choice. Please enter 1-4.")
                
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
