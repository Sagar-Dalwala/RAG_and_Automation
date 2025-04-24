from rag_utils import process_input, answer_question

# Test with sample text
text = 'LangChain is a framework for developing applications powered by language models. It enables applications that are context-aware, reason, and learn from feedback.'
vector_store = process_input('Text', text)
answer = answer_question(vector_store, 'What is LangChain?')
print('Question: What is LangChain?')
print('Answer:', answer) 