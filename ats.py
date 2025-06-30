# -*- coding: utf-8 -*-
"""ats
Original file is located at
    https://colab.research.google.com/drive/146gZ-rJOuoo2JLDLeKVWzzlMkNwTdIvD
"""

# Install dependencies (Colab only)
# Install dependencies (Colab only)
!pip install langchain langchain-community faiss-cpu sentence-transformers pypdf pandas

import pandas as pd

# Simulate JD dataset
jd_data = {
    'job_id': [101, 102, 103],
    'job_title': ['Data Scientist', 'ML Engineer', 'NLP Specialist'],
    'skills': ['Python, ML, SQL, TensorFlow',
               'Deep Learning, NLP, Python, AWS',
               'Transformers, LLM, Prompt Engineering']
}
jd_df = pd.DataFrame(jd_data)
print("Sample JD Dataset:")
print(jd_df)

from google.colab import files
uploaded = files.upload()
resume_path = list(uploaded.keys())[0]

from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader(resume_path)
pages = loader.load_and_split()
resume_text = " ".join([p.page_content for p in pages])

print("\nExtracted Resume Text:")
print(resume_text[:1500])

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
resume_emb = model.encode([resume_text])
jd_embs = model.encode(jd_df['skills'].tolist())

index = faiss.IndexFlatL2(resume_emb.shape[1])
index.add(np.array(jd_embs))
D, I = index.search(np.array(resume_emb), k=3)
matched_jobs = jd_df.iloc[I[0]]
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
resume_emb = model.encode([resume_text])
jd_embs = model.encode(jd_df['skills'].tolist())

index = faiss.IndexFlatL2(resume_emb.shape[1])
index.add(np.array(jd_embs))
D, I = index.search(np.array(resume_emb), k=3)
matched_jobs = jd_df.iloc[I[0]]

print("\nTop Matching Job Roles:")
print(matched_jobs[['job_id', 'job_title']])
