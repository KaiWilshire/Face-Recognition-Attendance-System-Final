To ensure the application runs correctly on a new device, follow these steps to set up the 
environment and the "AI brain" (.deepface). 
Step 1: Setting up the AI Model (.deepface) Because the system uses the VGG-Face model, it 
requires pre-trained "weights" to recognize faces. 
1. Locate the .deepface folder from the provided Google Drive link/USB. 
2. Navigate to your computer's User directory (usually C:\Users\YourUsername). 
3. Paste the entire .deepface folder here. 
• Note: If this folder is missing, the application will attempt to download 500MB of 
data on the first launch, which requires an internet connection. 
Step 2: Launching the Application 
1. Copy the application folder to your local drive. 
2. Double-click ai_school_system.exe to start. 
Step 3: Automatic Folder Creation Upon the first launch, the system will automatically create 
the following: 
• face_id_data/: This is where student photos will be stored in sub-folders. 
• attendance_log.csv: The database where all check-ins are recorded. 
• face_encodings.pkl: The "memory" file created after you register your first student. 
