# ClimateLens

Organizational Climate Assessment System: Thai employee-survey sentiment, topic categorization and Climate Index dashboard.

## Run the dashboard (mock data)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run dashboard/app.py
```

## Structure

- `dashboard/app.py`: entry point, navigation and global filters (months, departments, data source)
- `dashboard/views/`: Overview, Departments, Trends, Feedback Explorer, Upload & Analyze
- `dashboard/climate.py`: data access, filters, Climate Index (`50 + 50 × (pos − neg) / n`; ≥60 Healthy, 45–60 Watch, <45 Needs attention)
- `dashboard/mock_data.py`: generated Thai survey responses (Jan–Oct 2026, 7 departments, 7 categories)
- `dashboard/classifier.py`: keyword placeholder; replace `classify()` with the real sentiment model + LLM categorizer
