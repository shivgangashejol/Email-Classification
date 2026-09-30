# Automated Email Classification & Production Logging Pipeline
Engineered an end-to-end NLP pipeline utilizing custom text preprocessing, tokenization, and TF-IDF vectorization to isolate malicious entries. Designed database layer to log live inference requests, archiving tokenized metadata and real-time model predictions for secure audit trails.Optimized model hyperparameters to achieve a 97% F1-score, ensuring highly accurate real-time classification.

## Key Architectural Features
* **Advanced NLP Tokenization Engine:** Implements structural regex profiling to isolate high-risk verbs, indicators (!, $), and automatically masks hyperlinks and email addresses into uniform analytical structures ([URL], [EMAIL]).
* **Balanced Machine Learning Classifier:** Employs a hyperparameter-tuned Scikit-Learn Logistic Regression model backed by a custom-bound TF-IDF Vectorizer to offset class imbalances and target an optimized 97.00% F1-Score.
* **State-Driven Engine Asset Caching:** Leverages Streamlit resource caching (@st.cache_resource) to initialize vectorization matrices and model weights once, ensuring instantaneous real-time classification.
* **Persistent Compliance Audit Trail:** Implements an automated SQLite relational database bridge that serializes structural log metadata (JSON token fields, inputs, predictions, and confidence metrics) for security tracking.
  
## Repository & File System Structure
```bash
├── app.py           
├── pipeline.py      
├── pipeline_logs.db 
└── README.md       
```
## Installation & Quick Start
Follow these structural setup steps to clone, configure, and execute the production environment locally.
### 1. Prerequisite Infrastructure
Ensure you have Python 3.8+ installed on your machine along with the git CLI utility.

### 2. Clone the Repository
Open your native terminal instance and pull the codebase:
```bash
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
```
### 3. Install Target Production Dependencies
Install all core packages required to run the classification pipeline:
``` base
pip install streamlit pandas scikit-learn
```
### 4. Execute the Application Terminal Hub
Launch the local web server to initialize the pipeline assets and database structures:
``` base
streamlit run app.py
```

