# SlopeLab dashboard implementation requirements

## Files that must be in the same project folder
- `app.py` — the redesigned Streamlit dashboard.
- `slope_app.py` — the original SlopeLab calculation engine. This is required for real Bishop/Fellenius calculations and the critical slip-surface plot.
- `requirements.txt` — Python package dependencies.

## Install and run
```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Required behaviour
1. Open on a fresh, editable Worksheet 1; do not preload the old worked example.
2. Provide worksheet switching, New Worksheet and Delete controls. Each worksheet has its own geometry, soil, groundwater, settings and results.
3. Store worksheets in persistent storage. This starter uses a local SQLite database (`slopelab_workspaces.sqlite3`) beside `app.py`.
4. Recalculate only when Run Analysis is pressed. Never display made-up factor-of-safety values or a stable/unstable badge before a real calculation succeeds.
5. Show Bishop and Fellenius factors of safety, selected method, status interpretation, and the critical slip-surface plot when the calculation engine supplies them.
6. Show validation/errors clearly. Do not silently replace missing engineering inputs with fabricated values.
7. Keep the dark navy dashboard styling, blue worksheet navigation, teal New Worksheet and Run Analysis actions, and a right-side results panel.
8. For a cloud deployment, configure persistent database storage and user authentication. A local SQLite file may not persist on some ephemeral hosting platforms.

## Important integration note
The uploaded `worksheet_app.py` imports functions from `slope_app.py`. This redesign uses those same engine functions. The `slope_app.py` file was not included with the upload, so the UI can run in preview mode without it, but real analysis cannot run until the original engine file is added beside `app.py`.
