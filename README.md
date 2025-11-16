# Road Safety Intervention GPT 🚧

AI-powered system that recommends **IRC-based road safety interventions** using Retrieval-Augmented Generation (RAG), Groq's Llama 3.1, and FAISS vector search.

**Team MUFFIN** — Sneha Chakraborty & Divyansh Pathak  
*National Road Safety Hackathon 2025*

---

## 🎯 Features

- Intelligent query processing using semantic search
- Category-based retrieval for precise results
- IRC standards-based recommendations
- Fast inference powered by Groq
- Simple command-line interface

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Groq API Key ([Get one free](https://console.groq.com))

### Installation

1. Clone this repository:
```bash
git clone https://github.com/YOUR_USERNAME/road-safety-rag.git
cd road-safety-rag
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set your Groq API key:
```bash
# Linux/Mac
export GROQ_API_KEY="your_api_key_here"

# Windows (Command Prompt)
set GROQ_API_KEY=your_api_key_here

# Windows (PowerShell)
$env:GROQ_API_KEY="your_api_key_here"
```

4. Make sure your CSV file is in the root directory:
   - File name: `GPT_Input_DB(Sheet1).csv`

5. Run the app:
```bash
python app.py
```

---

## 📁 Project Structure

```
road-safety-rag/
├── app.py                          # CLI interface
├── rag_engine.py                   # RAG pipeline & logic
├── requirements.txt                # Python dependencies
├── GPT_Input_DB(Sheet1).csv        # Road safety database
├── faiss_indices_by_category/      # FAISS indices (auto-generated)
└── README.md                       # This file
```

---

## 🔧 How It Works

1. **Data Loading**: Reads road safety interventions from CSV
2. **Embedding Generation**: Creates vector embeddings using sentence-transformers
3. **FAISS Indexing**: Builds category-specific vector stores
4. **Query Processing**: 
   - Identifies most relevant category
   - Retrieves top-3 matching interventions
   - LLM generates contextualized recommendation
5. **Response**: Returns IRC-compliant intervention with codes and clauses

---

## 💡 Example Queries

- "Sharp curve on highway without warning signs"
- "Pedestrian crossing area lacking proper markings"
- "Poor visibility at intersection during night time"
- "Narrow bridge with heavy vehicle traffic"
- "School zone without speed limit signage"

---

## 🛠️ Technical Stack

- **LLM**: Groq (Llama 3.1-8B)
- **Embeddings**: sentence-transformers/all-mpnet-base-v2
- **Vector Store**: FAISS
- **Framework**: LangChain
- **Data Processing**: Pandas, NumPy

---

## 📊 CSV Format

Your CSV must have these columns:

- `S. No.` - Serial number
- `problem` - Description of road safety problem
- `category` - Problem category
- `type` - Type of intervention
- `data` - Detailed description
- `code` - IRC code reference
- `clause` - Specific clause number

---

## 🐛 Troubleshooting

**"GROQ_API_KEY environment variable not set!"**
- Set the environment variable before running

**"Missing column" error**
- Ensure CSV has all required columns

**Slow first run**
- First run builds FAISS indices (2-5 minutes)
- Subsequent runs use cached indices

**Dependencies not installing**
- Upgrade pip: `pip install --upgrade pip`
- Try: `pip install -r requirements.txt --no-cache-dir`

---

## 📄 License

MIT License

---

## 👥 Team MUFFIN

**Contributors:**
- [@Kweenbee187](https://github.com/Kweenbee187) - Sneha Chakraborty
- [@tituatgithub](https://github.com/tituatgithub) - Divyansh Pathak

*National Road Safety Hackathon 2025*

---

**Made with ❤️ for safer roads in India**
