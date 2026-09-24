from app.llm.groq_provider import GroqProvider


provider = GroqProvider()

response = provider.generate(
    system_prompt="You are a concise AI engineering assistant.",
    user_prompt="Explain in one sentence what RAG means.",
)

print("\nGROQ RESPONSE:")
print(response)