# 🧭 Assessment Recommendation System

Note: The streamlit app is available at : http://44.214.181.171/

### 🎯 Goal
The goal of this project is to build an **Assessment Recommender System** that suggests the most relevant **SHL assessments** based on a given **user query** or **job description (JD)**.

Users can either:
- Enter a **text query** describing the hiring requirement, **or**
- Provide a **URL** to a job description (HTML or PDF)

The system then recommends the **top 10 assessments** that best fit the job’s skills, level, and constraints.

---

Note: Jupyter Python file of all approaches tried to make project stored in notebooks folder.

###  Dataset Creation

Our dataset was created by **crawling the official SHL website**, where each product page represents one assessment.

- The **assessment fact sheets (PDFs)** were parsed using **PyMuPDF** / `pdfminer.six` to extract the core test content.
- Additional details such as duration, test type, job level, and adaptive support were parsed from the **HTML webpages** using **BeautifulSoup**.

---

#### Dataset Columns

The python file for data extraction available in Script Folder, with comments made for detailed explanation. 

Each dataset record represents one SHL assessment and contains the following fields:

| Column | Description |
|---------|-------------|
| `name` | Name of the assessment |
| `url` | Direct URL to the assessment on SHL’s site |
| `pdf_text` | Extracted text from the assessment’s fact sheet PDF |
| `description` | Short summary of what the assessment measures |
| `duration` | Average testing time (in minutes) |
| `job_level` | Target job level (e.g., entry-level, mid, senior) |
| `test_language` | Language in which the assessment is available |
| `adaptive_support` | Whether adaptive testing is supported |
| `remote_support` | Whether remote testing is supported |
| `test_type` | List of test categories (e.g., “Knowledge & Skills”, “Simulations”) |

Example record:
json
{
  "name": ".NET MVVM (New)",
  "url": "https://www.shl.com/products/product-catalog/view/net-mvvm-new/",
  "pdf_text": "...",
  "description": "Multi-choice test that measures the knowledge of MVVM pattern, scenarios, data validation, ViewModel communication and Quick-start.",
  "duration": 5,
  "job_level": "Mid-Professional, Professional Individual Contributor",
  "test_language": "English (US)",
  "adaptive_support": "No",
  "remote_support": "Yes",
  "test_type": ["Knowledge & Skills"]
}


### Approach

## Query Simplification (LLM Preprocessing)

We use Google Gemini (2.5 Flash) to extract structured details from the user query or JD, such as:

{
  "looking_for": "Python developer with strong data analysis and ML skills",
  "instructions": "Duration must be under 60 minutes, English language, one sitting",
  "job_level": "mid",
  "language": "English"
}

looking_for // skills or expertise are expected from JD or the query mentions directly or indirectly
intructions // instructions about time, langugae, max questions, combination of assessments etc
job_level // the job level of the candidate if mentioned in JD
language // the langugae the assessment in required in

## Hybrid Similarity Matching (TF-IDF + Embeddings)

We combine two techniques to compute similarity between the user’s simplified query and all assessments:

SentenceTransformer embeddings for semantic similarity (all-MiniLM-L6-v2)

TF-IDF for lexical (keyword-based) similarity

final_score = 0.75 * emb_looking_for + 0.10 * emb_job_level + 0.10 * emb_language + 0.15 * tfidf_looking_for

## LLM Reranking for Context-Aware Selection

To ensure results match user constraints (like total duration or test type), the top 25 assessments are passed to Gemini again.

The LLM reranker picks the best 10 assessments considering:

Topic and skills relevance

Job level and language

Constraints (e.g., “total test time ≤ 1 hour”)

Diversity (avoid near-duplicates)

### Example

User Query:

“Need a mid-level Python developer test, all assessments together should take about 1 hour.”

System Flow:

Query summarized via Gemini.

Similarity computed using TF-IDF + SentenceTransformer.

Gemini reranks top 25 results to choose the 10 best assessments matching the “1 hour” constraint.

### Deployment

The complete system is deployed using Streamlit, FastAPI, and Docker on AWS EC2.

Components:
Component	Description
FastAPI (Backend)	Provides the /api/recommend endpoint that performs all recommendation logic
Streamlit (Frontend)	User interface to input either free text or a job description URL
Nginx (Proxy)	Routes traffic between the frontend and backend
Docker Compose	Manages multi-container setup

The Streamlit app is accessible at:: http://44.214.181.171/
