# Road Safety Intervention GPT 🚧

AI-powered system that recommends **IRC-based road safety interventions** using Retrieval-Augmented Generation (RAG), Groq's Llama 3.1, and FAISS vector search.

**Team MUFFIN** — Sneha Chakraborty & Divyansh Pathak  
*National Road Safety Hackathon 2025*

---

## 🎯 Features

- **Intelligent Query Processing**: Uses semantic search to find relevant interventions
- **Category-Based Retrieval**: Automatically categorizes queries for precise results
- **IRC Standards**: All recommendations based on Indian Roads Congress guidelines
- **Fast Inference**: Powered by Groq's lightning-fast LLM infrastructure
- **User-Friendly Interface**: Clean Gradio UI for easy interaction

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Groq API Key ([Get one here](https://console.groq.com))

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
export GROQ_API_KEY="your_api_key_here"
```

Or create a `.env` file:
```
GROQ_API_KEY=your_api_key_here
```

4. Place your CSV file:
- Ensure `GPT_Input_DB(Sheet1).csv` is in the root directory

5. Run the app:
```bash
python app.py
```

6. Open your browser to the URL shown (usually `http://127.0.0.1:7860`)

---

## 📁 Project Structure

```
road-safety-rag/
├── app.py                          # Gradio frontend
├── rag_engine.py                   # RAG pipeline & logic
├── requirements.txt                # Python dependencies
├── GPT_Input_DB(Sheet1).csv        # Road safety database
├── faiss_indices_by_category/      # Generated FAISS indices (auto-created)
└── README.md                       # This file
```

---

## 🔧 How It Works

1. **Data Loading**: Reads road safety interventions from CSV database
2. **Embedding Generation**: Creates vector embeddings using sentence-transformers
3. **FAISS Indexing**: Builds category-specific vector stores for fast retrieval
4. **Query Processing**: 
   - User submits a road safety issue
   - System identifies the most relevant category
   - Retrieves top-k matching interventions
   - LLM generates a contextualized recommendation
5. **Response**: Returns IRC-compliant intervention with codes and clauses

---

## 💡 Example Queries

Try these sample queries:

- "Sharp curve on highway without warning signs"
- "Pedestrian crossing area lacking proper markings"
- "Poor visibility at intersection during night time"
- "Narrow bridge with heavy vehicle traffic"
- "School zone without speed limit signage"

---

## 🛠️ Technical Stack

| Component | Technology |
|-----------|-----------|
| **LLM** | Groq (Llama 3.1-8B) |
| **Embeddings** | sentence-transformers/all-mpnet-base-v2 |
| **Vector Store** | FAISS |
| **Framework** | LangChain |
| **Frontend** | Gradio |
| **Data Processing** | Pandas, NumPy |

---

## 📊 CSV Format

Your CSV should have these columns:

- `S. No.` - Serial number
- `problem` - Description of the road safety problem
- `category` - Problem category
- `type` - Type of intervention
- `data` - Detailed description
- `code` - IRC code reference
- `clause` - Specific clause number

---

## 🔒 Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GROQ_API_KEY` | Your Groq API key | Yes |

---

## 🐛 Troubleshooting

### "Missing GROQ_API_KEY"
- Set the environment variable: `export GROQ_API_KEY="your_key"`

### "Missing column" error
- Ensure your CSV has all required columns (see CSV Format section)

### Slow first run
- First run builds FAISS indices (2-5 minutes)
- Subsequent runs use cached indices and are much faster

### Dependencies not installing
- Upgrade pip: `pip install --upgrade pip`
- Try: `pip install -r requirements.txt --no-cache-dir`

---

## 📈 Performance

- **Embedding Model**: 768-dimensional vectors
- **Retrieval**: Top-3 most relevant documents per query
- **Response Time**: ~2-3 seconds per query (after initialization)
- **Accuracy**: Category-based filtering ensures high precision

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push and create a Pull Request

---

## 📄 License

MIT License - feel free to use for your projects!

---

## 👥 Team MUFFIN

**Sneha Chakraborty** & **Divyansh Pathak**

*National Road Safety Hackathon 2025*

---

## 🙏 Acknowledgments

- Indian Roads Congress (IRC) for road safety standards
- Groq for fast LLM inference
- LangChain for RAG framework
- HuggingFace for embeddings models

---

## 📧 Contact

For questions or support, reach out to Team MUFFIN!

---

**Made with ❤️ for safer roads in India**
