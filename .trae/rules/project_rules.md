always run backend using:
uvicorn main:app --host 0.0.0.0 --port 8000
and not python file directly

don't use curl commands always use:
Invoke-RestMethod -Uri ... command