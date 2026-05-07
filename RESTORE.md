# Restore Points

Restore archives are stored in the `RestorePoints` folder.

To restore one:

1. Stop the Flask server if it is running.
2. Extract the chosen `.zip` archive.
3. Copy the extracted files back into this project folder.
4. Reinstall dependencies if needed:
   ```powershell
   cd Deploy
   ..\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```
5. Start the app:
   ```powershell
   cd Deploy
   ..\.venv\Scripts\python.exe run_server.py
   ```

The restore archives intentionally exclude `.venv`, datasets, caches, generated logs, and uploaded videos.
