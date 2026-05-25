from google import genai

client = genai.Client(api_key="AIzaSyCP2n_jffkdwi92uzYFdPqCaFKO_xXFG-Q")

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="How to treat bacterial spot in bell pepper plants?"
)

print(response.text)