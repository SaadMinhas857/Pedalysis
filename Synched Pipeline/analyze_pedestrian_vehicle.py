import pandas as pd
from datetime import datetime
import os

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

def analyze_pedestrian_vehicle():
    # Read the CSV files with correct paths
    pedestrian_df = pd.read_csv(os.path.join('logs', 'pedestrian_analysis_20250414_134143.csv'))
    vehicle_df = pd.read_csv(os.path.join('logs', 'vehicle_analysis_20250414_134143.csv'))
    
    results = []
    
    # Process each pedestrian
    for _, pedestrian in pedestrian_df.iterrows():
        pedestrian_id = pedestrian['Pedestrian_ID']
        
        # Get time ranges for each lane
        lane_ranges = {}
        
        # Process each lane
        for lane in [1, 2, 3]:
            line1_time = time_to_seconds(pedestrian[f'Lane{lane}_Line1_Time'])
            line2_time = time_to_seconds(pedestrian[f'Lane{lane}_Line2_Time'])
            line3_time = time_to_seconds(pedestrian[f'Lane{lane}_Line3_Time'])
            
            if line1_time is not None and line2_time is not None and line3_time is not None:
                # Determine direction by comparing line1 and line3 times
                moving_left_to_right = line1_time > line3_time
                
                # Adjust times based on direction
                if moving_left_to_right:
                    line1_time -= 2  # Subtract 2 seconds from first line
                    line3_time += 2  # Add 2 seconds to last line
                else:
                    line1_time += 2  # Add 2 seconds to first line
                    line3_time -= 2  # Subtract 2 seconds from last line
                
                lane_ranges[lane] = {
                    'start': line1_time,
                    'middle': line2_time,
                    'end': line3_time,
                    'direction': 'left_to_right' if moving_left_to_right else 'right_to_left'
                }
        
        # Check each vehicle
        for _, vehicle in vehicle_df.iterrows():
            lane = vehicle['Lane_Number']
            vehicle_time = time_to_seconds(vehicle['Detection_Time'])
            
            # Check if vehicle time falls within pedestrian's range for this lane
            if lane in lane_ranges:
                # Handle both forward and reverse time ranges
                time_min = min(lane_ranges[lane]['start'], lane_ranges[lane]['end'])
                time_max = max(lane_ranges[lane]['start'], lane_ranges[lane]['end'])
                
                if (time_min <= vehicle_time <= time_max):
                    # Calculate time difference between vehicle and pedestrian's middle time
                    time_diff_seconds = vehicle_time - lane_ranges[lane]['middle']
                    
                    # Format time difference for display
                    if time_diff_seconds == 0:
                        time_diff_str = '<1'
                    else:
                        time_diff_str = str(abs(time_diff_seconds))
                    
                    # For display purposes, show 'right_to_left' when moving 'left_to_right'
                    display_direction = 'right_to_left' if lane_ranges[lane]['direction'] == 'left_to_right' else 'left_to_right'
                    
                    results.append({
                        'Pedestrian_ID': pedestrian_id,
                        'Vehicle_ID': vehicle['Vehicle_ID'],
                        'Lane': lane,
                        'Vehicle_Time': vehicle['Detection_Time'],
                        'Pedestrian_Middle_Time': seconds_to_time(lane_ranges[lane]['middle']),
                        'Time_Difference': time_diff_str,
                        'Vehicle_Type': vehicle['Vehicle_Type'],
                        'Direction': display_direction
                    })
    
    # Convert results to DataFrame and save to CSV
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        # Sort results by Pedestrian_ID and Time_Difference (convert '<1' to 0 for sorting)
        results_df['Sort_Time'] = results_df['Time_Difference'].apply(lambda x: 0 if x == '<1' else int(x))
        results_df = results_df.sort_values(['Pedestrian_ID', 'Sort_Time'])
        results_df = results_df.drop('Sort_Time', axis=1)
        
        # Create results directory if it doesn't exist
        os.makedirs('results', exist_ok=True)
        
        # Save detailed results
        output_file = os.path.join('results', 'pedestrian_vehicle_analysis.csv')
        results_df.to_csv(output_file, index=False)
        print(f"Detailed analysis saved to '{output_file}'")
        
        # Create simplified summary
        summary_data = []
        for _, row in results_df.iterrows():
            # Convert time difference to number
            time_diff = row['Time_Difference']
            
            summary_data.append({
                'Pedestrian_ID': row['Pedestrian_ID'],
                'Vehicle_ID': row['Vehicle_ID'],
                'Vehicle_Class': row['Vehicle_Type'],
                'Vehicle_Lane': row['Lane'],
                'Time_Difference(s)': time_diff
            })
        
        # Save summary to CSV
        summary_df = pd.DataFrame(summary_data)
        summary_file = os.path.join('results', 'pedestrian_vehicle_summary.csv')
        summary_df.to_csv(summary_file, index=False)
        print(f"Summary saved to '{summary_file}'")
        
        # Print summary
        print("\nSummary Results:")
        print(summary_df.to_string(index=False))
    else:
        print("No matching vehicle-pedestrian pairs found.")

if __name__ == "__main__":
    analyze_pedestrian_vehicle() 