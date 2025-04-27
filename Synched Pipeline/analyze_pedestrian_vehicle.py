import pandas as pd
from datetime import datetime
import os
import glob
from PCA_PS_Pipeline import PedestrianPipeline
import sqlite3
import mysql.connector

def time_to_seconds(time_str):
    if pd.isna(time_str):
        return None
    h, m, s = map(int, time_str.split(':'))
    return h * 3600 + m * 60 + s

def seconds_to_time(seconds):
    if seconds is None:
        return None
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def get_latest_files():
    """Get the most recent pedestrian and vehicle analysis files"""
    log_dir = 'Synched Pipeline/logs'
    
    # Get all pedestrian and vehicle analysis files
    ped_files = glob.glob(os.path.join(log_dir, 'pedestrian_analysis_*.csv'))
    vehicle_files = glob.glob(os.path.join(log_dir, 'vehicle_analysis_*.csv'))
    
    if not ped_files or not vehicle_files:
        print("No analysis files found in the logs directory")
        return None, None
    
    # Get the most recent files based on timestamp in filename
    latest_ped = max(ped_files, key=os.path.getctime)
    latest_vehicle = max(vehicle_files, key=os.path.getctime)
    
    print(f"Using pedestrian file: {os.path.basename(latest_ped)}")
    print(f"Using vehicle file: {os.path.basename(latest_vehicle)}")
    
    return latest_ped, latest_vehicle

def analyze_pedestrian_vehicle():
    """Analyze pedestrian and vehicle interactions after CSV files are finalized"""
    try:
        # Get the most recent files
        ped_file, vehicle_file = get_latest_files()
        
        if not (ped_file and vehicle_file):
            print("Error: Required CSV files not found")
            return
            
        print(f"Processing pedestrian file: {ped_file}")
        print(f"Processing vehicle file: {vehicle_file}")
        
        # Read the CSV files
        ped_df = pd.read_csv(ped_file)
        vehicle_df = pd.read_csv(vehicle_file)
        
        # Initialize database connection for parking status
        db_connection = mysql.connector.connect(**DB_CONFIG)
        db_cursor = db_connection.cursor()
        
        try:
            # First, process parking status for all pedestrians
            print("\nProcessing parking status for all pedestrians...")
            for _, pedestrian in ped_df.iterrows():
                ped_id = pedestrian['Pedestrian_ID']
                
                # Get parking status, default to 0 if not present
                parking_status = 0.0
                if 'Parking' in pedestrian:
                    try:
                        parking_status = float(pedestrian['Parking'])
                        print(f"Found parking status {parking_status} for pedestrian {ped_id}")
                    except (ValueError, TypeError) as e:
                        print(f"Error converting parking status for pedestrian {ped_id}: {e}")
                        parking_status = 0.0
                
                # Insert parking status (behavior_id 15)
                try:
                    print(f"Inserting parking status {parking_status} for pedestrian {ped_id}")
                    db_cursor.execute("""
                        INSERT INTO scene_behavior_feature
                        (scene_key, object_type, behavior_id, behavior_value)
                        VALUES (%s, %s, %s, %s)
                        ON DUPLICATE KEY UPDATE behavior_value = VALUES(behavior_value)
                    """, (int(ped_id), 'pedestrian', 15, float(parking_status)))
                    db_connection.commit()
                    print(f"Successfully inserted parking status for pedestrian {ped_id}")
                except Exception as e:
                    print(f"Error inserting parking status for pedestrian {ped_id}: {e}")
                    db_connection.rollback()
            
            # Initialize results DataFrame
            results = []
            
            # Process each pedestrian for vehicle interactions
            for _, pedestrian in ped_df.iterrows():
                ped_id = pedestrian['Pedestrian_ID']
                direction = pedestrian['Direction']
                lane_ranges = {}
                
                # Get parking status for this pedestrian
                parking_status = 0.0
                if 'Parking' in pedestrian:
                    try:
                        parking_status = float(pedestrian['Parking'])
                    except (ValueError, TypeError):
                        parking_status = 0.0
                
                # Get time ranges for each lane this pedestrian crosses
                for lane in range(1, 4):  # Assuming 3 lanes
                    start_col = f'Lane{lane}_Start'
                    end_col = f'Lane{lane}_End'
                    
                    if start_col in pedestrian and end_col in pedestrian:
                        start_time = time_to_seconds(pedestrian[start_col])
                        end_time = time_to_seconds(pedestrian[end_col])
                        
                        if not pd.isna(start_time) and not pd.isna(end_time):
                            middle_time = (start_time + end_time) / 2
                            lane_ranges[lane] = {
                                'start': start_time,
                                'end': end_time,
                                'middle': middle_time,
                                'direction': direction
                            }
                
                # Check each vehicle
                for _, vehicle in vehicle_df.iterrows():
                    lane = vehicle['Lane_Number']
                    vehicle_id = vehicle['Vehicle_ID']
                    vehicle_time = time_to_seconds(vehicle['Detection_Time'])
                    vehicle_type = vehicle['Vehicle_Type']
                    
                    # Check if vehicle time falls within pedestrian's range for this lane
                    if lane in lane_ranges:
                        # Handle both forward and reverse time ranges
                        time_min = min(lane_ranges[lane]['start'], lane_ranges[lane]['end'])
                        time_max = max(lane_ranges[lane]['start'], lane_ranges[lane]['end'])
                        
                        if time_min <= vehicle_time <= time_max:
                            # Calculate time difference between vehicle and pedestrian's middle time
                            time_diff_seconds = abs(vehicle_time - lane_ranges[lane]['middle'])
                            
                            # Format time difference for display
                            time_diff_str = '<1' if time_diff_seconds == 0 else str(time_diff_seconds)
                            
                            # Store the result with parking status
                            results.append({
                                'Pedestrian_ID': ped_id,
                                'Vehicle_ID': vehicle_id,
                                'Time_Difference': time_diff_str,
                                'Vehicle_Type': vehicle_type,
                                'Lane_Number': lane,
                                'Direction': direction,
                                'Parking_Status': parking_status  # Add parking status to results
                            })
            
            # Create results DataFrame and save to CSV
            if results:
                results_df = pd.DataFrame(results)
                output_file = 'results/pedestrian_vehicle_analysis.csv'
                os.makedirs('results', exist_ok=True)
                results_df.to_csv(output_file, index=False)
                print(f"\nAnalysis results saved to {output_file}")
                
                # Now that CSV is finalized, update the database with PSM results
                update_database_with_psm_results(results_df)
            else:
                print("\nNo interactions found between pedestrians and vehicles")
                
        finally:
            # Close database connection
            if db_cursor:
                db_cursor.close()
            if db_connection:
                db_connection.close()
            
    except Exception as e:
        print(f"\nError analyzing pedestrian-vehicle interactions: {e}")
        import traceback
        traceback.print_exc()

def update_database_with_psm_results(results_df):
    """Update database with PSM results after CSV is finalized"""
    try:
        # Initialize database connection
        db_connection = mysql.connector.connect(**DB_CONFIG)
        db_cursor = db_connection.cursor()
        
        print("\nStarting database updates...")
        
        # Process each unique pedestrian
        for pedestrian_id in results_df['Pedestrian_ID'].unique():
            ped_data = results_df[results_df['Pedestrian_ID'] == pedestrian_id].iloc[0]
            
            # Get values
            time_diff = ped_data['Time_Difference']
            vehicle_type = ped_data['Vehicle_Type']
            direction = ped_data['Direction']
            
            # Convert time_diff to float, handling '<1' case
            time_diff_value = 0.5 if time_diff == '<1' else float(time_diff)
            
            try:
                # Insert PSM time difference (behavior_id 14)
                db_cursor.execute("""
                    INSERT INTO scene_behavior_feature
                    (scene_key, object_type, behavior_id, behavior_value)
                    VALUES (%s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE behavior_value = VALUES(behavior_value)
                """, (pedestrian_id, 'pedestrian', 14, time_diff_value))
                print(f"\nInserted behavior_id 14 (PSM time) for pedestrian {pedestrian_id}: {time_diff_value}")
                
                # Insert vehicle type (behavior_id 13)
                vehicle_type_id = get_vehicle_type_id(vehicle_type)
                db_cursor.execute("""
                    INSERT INTO scene_behavior_feature
                    (scene_key, object_type, behavior_id, behavior_value)
                    VALUES (%s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE behavior_value = VALUES(behavior_value)
                """, (pedestrian_id, 'pedestrian', 13, float(vehicle_type_id)))
                print(f"Inserted behavior_id 13 (vehicle type) for pedestrian {pedestrian_id}: {vehicle_type_id}")
                
                # Insert direction (behavior_id 11) - FIXED
                direction_value = 1.0 if direction.lower() in ['right_to_left', 'forward', 'positive'] else 0.0
                print(f"Processing direction: {direction} -> {direction_value}")
                db_cursor.execute("""
                    INSERT INTO scene_behavior_feature
                    (scene_key, object_type, behavior_id, behavior_value)
                    VALUES (%s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE behavior_value = VALUES(behavior_value)
                """, (int(pedestrian_id), 'pedestrian', 11, float(direction_value)))
                print(f"Inserted behavior_id 11 (direction) for pedestrian {pedestrian_id}: {direction_value}")
                
                # Commit after each pedestrian
                db_connection.commit()
                print(f"Committed changes for pedestrian {pedestrian_id}")
                
            except Exception as e:
                print(f"\nError updating database for pedestrian {pedestrian_id}:")
                print(f"Error details: {str(e)}")
                print("Current values being processed:")
                print(f"- Direction: {direction} -> {direction_value}")
                db_connection.rollback()
                continue
        
        print("\nAll database updates completed successfully")
        
    except Exception as e:
        print(f"\nError in database update process: {e}")
        if 'db_connection' in locals():
            db_connection.rollback()
        import traceback
        traceback.print_exc()
    finally:
        if 'db_cursor' in locals() and db_cursor:
            db_cursor.close()
        if 'db_connection' in locals() and db_connection:
            db_connection.close()
        print("Database connection closed")

def get_vehicle_type_id(vehicle_type):
    """Convert vehicle type string to numeric ID"""
    vehicle_type_map = {
        'Biker': 1,
        'Motorbike': 2,
        'Car': 3,
        'Taxi': 4,
        'Bus': 5,
        'Truck': 6,
        'Unknown': 0
    }
    return vehicle_type_map.get(vehicle_type, 0)

if __name__ == "__main__":
    analyze_pedestrian_vehicle() 