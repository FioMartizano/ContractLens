# ContractLens

## Run locally

### Backend (Windows PowerShell)

Use Python 3.12. Do not run the backend with Python 3.15.

```powershell
cd ..\backend

py -3.12 -m venv .venv

Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python -m uvicorn main:app --reload --port 8000
```

The terminal prompt should begin with `(.venv)` after activation. The backend
should remain running and serve `http://127.0.0.1:8000/api/health`.

If an existing `.venv` is incomplete or has dependency conflicts, deactivate
and recreate it from the `backend` directory:

```powershell
deactivate
Remove-Item -Recurse -Force .venv
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m uvicorn main:app --reload --port 8000
```

Do not use `python -m pip` before activating the virtual environment unless
the command explicitly uses `.\.venv\Scripts\python.exe`.

### Frontend

Install the frontend dependencies and start Vite:

```sh
npm install
npm run dev
```

In another terminal, use Python 3.12, add your own `GEMINI_API_KEY` to
`backend/.env` (copy `backend/.env.example`), install the backend dependencies,
and start the API:

```sh
cd ../backend
python -m pip install -r requirements.txt
python -m uvicorn main:app --reload
```

The frontend sends uploaded PDF and image files to `http://localhost:8000` by
default. Set `VITE_API_URL` in a frontend `.env` file to use a different API
origin.

Uploads are extracted, analyzed with Gemini, checked against a seeded standard
clause library using ChromaDB retrieval, and saved in the persistent ChromaDB
collections. The database is stored under `backend/chroma_db` by default;
`CHROMA_PERSIST_DIRECTORY` can override that location. Use `POST /api/search`
with a JSON body such as `{"query":"payment terms","top_k":5}` to retrieve
matching contract sections. `POST /api/compare` compares two analyzed document
IDs. AI analysis is informational and is not legal advice.

Contract text is sent to the Gemini API configured by the backend key and is
also stored locally in ChromaDB. Do not upload documents unless you are
authorized to share their contents with that provider.
ChromaDB downloads its default local embedding model when the first document is
indexed, so the backend needs internet access for initial setup.
