api: uvicorn prediction_app:app --reload --host=0.0.0.0 --port=${PORT:-5000}
web: sh setup_streamlit.sh && streamlit run dashboard.py