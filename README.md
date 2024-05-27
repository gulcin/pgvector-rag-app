# pgvector-rag
An application to demonstrate how can you make a RAG using pgvector and PostgreSQL

## Requirements
- Python3
- PostgreSQL
- pgvector

## Install

Clone the repository

```
git clone git@github.com:gulcin/pgvector-rag.git
cd pgvector-rag
```

Install Dependencies

```
virtualenv env -p `which python`
source env/bin/activate
pip install -r requirements.txt
```

Add your .env variable

```
cp .env-example .env
```

## Run

```
python app.py --help

usage: app.py [-h] {create-db,import-data,chat} ...

Application Description

options:
  -h, --help            show this help message and exit

Subcommands:
  {create-db,import-data,chat}
                        Display available subcommands
    create-db           Create a database
    import-data         Import data
    chat                Use chat feature
```

## Run UI 

We use Streamlit for creating a simple Graphical User Interface for our pgvector-rag app. 

To be able to run Streamlit please do the following:

```
pip install streamlit
```

**Add keys/secrets to Streamlit secrets**

If you need to store secrets that Streamlit app will use, you can do this by creating
`.streamlit/secrets.toml` file under Streamlit directory and adding lines like following:

```
# .streamlit/secrets.toml
OPENAI_API_KEY = "YOUR_API_KEY"
```
**Run Streamlit app for generating UI**

```
streamlit run chatgptui.py
```
You can create as many apps you'd like and place them under Streamlit directory,
edit the keys if needed and run them like described above. 





