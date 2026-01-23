from src.ai_core.store import add_text_to_base
from src.ai_core.chat import ask_ai

# #fake data
# data = [
#     "Binbyte Technologies was founded by francis Kwame Sewor",
#     "Fastapi is a modern, fast web framework for building api's with python",
#     "the user is a senior computer engineering student at ashesi University"
# ]

# try:
#     add_text_to_base(data)
#     print("SUCCESS: Data ingested successfully")
# except Exception as e:
#     print(f"ERROR {e}")
    
print("--- Test 1 ---")
q1 = "Who founded Binbyte Technologies?"
print(f"Question: {q1}")
print(f"Answer: {ask_ai(q1)}")

print("\n--- Test 2 ---")
q2 = "What framework is used for APIs?"
print(f"Question: {q2}")
print(f"Answer: {ask_ai(q2)}")

print("\n--- Test 3 (Irrelevant Question)---")
q3 = "How do I bake a cake?"
print(f"Question: {q3}")
print(f"Answer: {ask_ai(q3)}")