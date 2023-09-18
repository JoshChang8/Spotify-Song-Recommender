# FROM continuumio/miniconda3
FROM python:3.8
# EXPOSE 8501
EXPOSE 8080
WORKDIR /workdir/
COPY . .
# COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
# HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health
# CMD streamlit run Get_Started.py --server.port $PORT
ENTRYPOINT ["streamlit", "run", "Get_Started.py", "--server.port=8080", "--server.address=0.0.0.0"]
