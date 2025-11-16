# app.py - Simple CLI interface for Road Safety RAG

from rag_engine import run_query

def main():
    print("\n" + "="*60)
    print("🚧 Road Safety Intervention GPT")
    print("Team MUFFIN — Sneha Chakraborty & Divyansh Pathak")
    print("="*60 + "\n")
    
    print("System is loading... (this may take a minute on first run)\n")
    
    while True:
        print("\n" + "-"*60)
        query = input("\n📝 Describe the road safety issue (or 'quit' to exit):\n> ")
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Thank you for using Road Safety GPT!")
            break
            
        if not query.strip():
            print("⚠️  Please enter a valid issue description.")
            continue
        
        print("\n🔍 Analyzing issue and finding interventions...\n")
        
        try:
            response = run_query(query)
            print("\n" + "="*60)
            print("💡 RECOMMENDATION:")
            print("="*60)
            print(response)
            print("="*60)
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            print("Please check your GROQ_API_KEY and try again.")

if __name__ == "__main__":
    main()
