## Installation & Setup
### Clone the repository:
- git clone https://github.com/HotWalterInYourArea/nir_bakalavra
- cd nir_bakalavra
### Start the infrastructure (Neo4j & Ollama):
- docker compose up -d neo4j ollama
### Pull the required LLM models:

- docker exec -it ollama-llm ollama pull llama3.1
- docker exec -it ollama-llm ollama pull qwen2.5:7b
- docker exec -it ollama-llm ollama pull llama3.2:3b
### Build the Python App container:

- docker compose up -d --build app
### Initialize the Graph (Load DBpedia data):
This will fetch entities, load them into Neo4j, and generate embeddings.

- docker exec -it gcr-experiment-app python dbpedia_loader.py
### Generate the QA Dataset:

- docker exec -it gcr-experiment-app python eval_dataset.py
## Running the Experiments
### 1. Baseline vs. GCR (Full Run):
This runs both pipelines and evaluates them using RAGAS. (Warning: This can take several hours on CPU).

- docker exec -d gcr-experiment-app python run_experiment.py > experiment.log 2>&1
- docker exec -it gcr-experiment-app tail -f experiment.log
### 2. Hyperparameter Sweep (Fast Run):
This runs the GCR filter with different parameters (alpha, beta, gamma, threshold) without calling the LLM, to see how they affect acceptance rates.

- docker exec -it gcr-experiment-app python run_gcr_sweep.py
### 3. Generate Plots & Tables:

- docker exec -it gcr-experiment-app python visualizer.py
  
Check the results/figures/ folder for the generated charts.
