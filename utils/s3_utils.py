import boto3
import os
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

class S3Manager:
    def __init__(self):
        # Streamlit Cloud에서는 st.secrets 사용, 로컬에서는 환경변수 사용
        self.aws_access_key_id = st.secrets["AWS_ACCESS_KEY_ID"] if "AWS_ACCESS_KEY_ID" in st.secrets else os.getenv("AWS_ACCESS_KEY_ID")
        self.aws_secret_access_key = st.secrets["AWS_SECRET_ACCESS_KEY"] if "AWS_SECRET_ACCESS_KEY" in st.secrets else os.getenv("AWS_SECRET_ACCESS_KEY")
        self.region = st.secrets["AWS_DEFAULT_REGION"] if "AWS_DEFAULT_REGION" in st.secrets else os.getenv("AWS_DEFAULT_REGION")
        self.bucket = st.secrets["S3_BUCKET_NAME"] if "S3_BUCKET_NAME" in st.secrets else os.getenv("S3_BUCKET_NAME")
        
        self.s3 = boto3.client(
            's3',
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            region_name=self.region
        )

    @st.cache_resource
    def download_db_if_needed(self, local_path: str = "data/temp_vector_db/chroma.sqlite3") -> bool:
        """벡터 DB를 S3에서 다운로드 (캐시 사용)"""
        try:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            if not os.path.exists(local_path):
                self.s3.download_file(self.bucket, "vector_db/chroma.sqlite3", local_path)
                st.success("Vector DB downloaded successfully!")
            return True
        except Exception as e:
            st.error(f"Error downloading DB: {str(e)}")
            return False

    def upload_db(self, local_path: str = "data/temp_vector_db/chroma.sqlite3") -> bool:
        """벡터 DB를 S3에 업로드"""
        try:
            if os.path.exists(local_path):
                self.s3.upload_file(local_path, self.bucket, "vector_db/chroma.sqlite3")
                return True
            return False
        except Exception as e:
            st.error(f"Error uploading DB: {str(e)}")
            return False 