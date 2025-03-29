from sqlalchemy import create_engine, text
import pymysql

def insert_sample_data():
    # Preset database connection details
    db_user = "Traffic1"
    db_password = "pedestrian"
    db_host = "localhost"
    db_name = "traffic_db"
    db_port = "3306"
    
    try:
        # Create connection string
        connection_string = f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        engine = create_engine(connection_string)
        
        # Test connection and insert data
        with engine.connect() as connection:
            print("✓ Database connection successful!")
            
            # Insert queries for each table
            queries = [
                """INSERT INTO behavior_feature (behavior_id, behavior_feature) 
                   VALUES (1, 'crossing');""",
                
                """INSERT INTO location_dimension 
                   (location_key, metro_city_province, district, neighborhood, spot)
                   VALUES (1, 'Lahore', 'Central', 'Gulberg', 'Main Boulevard');""",
                
                """INSERT INTO road_character_dimension 
                   (road_key, road_type, road_feature)
                   VALUES (1, 'arterial', 'intersection');""",
                
                """INSERT INTO scene_dimension 
                   (scene_key, object_type, event_type)
                   VALUES (1, 'pedestrian', 'crossing');""",
                
                """INSERT INTO time_dimension 
                   (time_key, week, day, day_night, hour)
                   VALUES (1, 'Week1', 'Monday', 'Day', '08:00');""",
                
                """INSERT INTO scene_behavior_feature 
                   (scene_key, object_type, behavior_id, behavior_value)
                   VALUES (1, 'pedestrian', 1, 0.85);""",
                
                """INSERT INTO fact_table 
                   (time_key, location_key, road_key, scene_key, scene_ratio, video_code)
                   VALUES (1, 1, 1, 1, 0.75, 'VID001');"""
            ]
            
            # Execute each query
            for query in queries:
                try:
                    connection.execute(text(query))
                    print(f"✓ Successfully executed: {query[:50]}...")
                except Exception as e:
                    print(f"Error executing query: {query[:50]}...")
                    print(f"Error message: {str(e)}")
            
            # Commit the transactions
            connection.commit()
            print("\n✓ All sample data inserted successfully!")
            
    except Exception as e:
        print(f"Connection error: {str(e)}")

if __name__ == "__main__":
    insert_sample_data() 