# ⚡ VULCAN Industrial OS

**A Cyber-Physical Intelligence System for Predictive Maintenance & Supply Chain Orchestration.**

VULCAN is an Agentic AI platform that bridges the gap between factory floor sensor data, technical repair documentation, and global commodity markets. It transforms reactive maintenance into proactive strategy.

## 🚀 Key Features

* **🔮 Predictive Machine Learning:** A local Random Forest model trained on 50,000+ rows of industrial telemetry to predict failure probabilities.
* **🧠 Agentic RAG:** Uses a local ChromaDB Vector Database to index and query 100+ page technical manuals (e.g., WEG Motors) for grounded repair instructions.
* **🌐 Supply Chain Intelligence:** Integrates real-time web search (Tavily) to monitor commodity prices (Copper, Aluminum) and global shipping disruptions.
* **📊 Modern Glassmorphism UI:** Built with Streamlit, featuring a dark-themed, high-contrast dashboard for executive decision-making.

## 🛠️ Technology Stack

* **Compute / Brain:** NVIDIA RTX 3050 Ti (Local Embeddings) / Google Gemini 3 Flash (LLM)
* **Orchestration:** LangChain / LangGraph (Multi-Tool Agent)
* **Data Engineering:** SQLite, Pandas, Scikit-learn
* **Frontend:** Streamlit with Custom CSS

## ⚙️ Installation & Setup

### Option 1: Standard Python Setup
1. Clone the repository: `git clone https://github.com/yourusername/vulcan-industrial-os.git`
2. Create a virtual environment: `python -m venv venv`
3. Install dependencies: `pip install -r requirements.txt`
4. Set up your `.env` file with `GOOGLE_API_KEY` and `TAVILY_API_KEY`.
5. Run the DB and ML setup:
   ```bash
   python src/tools/db_setup.py
   python src/tools/train_model.py