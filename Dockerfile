# FROM continuumio/miniconda3
FROM python:3.9-slim
# EXPOSE 8501
WORKDIR /Users/joshuachang/Development/SpotifyRecommender
COPY . .
# COPY requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt
EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health
CMD streamlit run Get_Started.py --server.port $PORT
# ENTRYPOINT ["streamlit", "run", "Get_Started.py", "--server.port=8501", "--server.address=0.0.0.0"]