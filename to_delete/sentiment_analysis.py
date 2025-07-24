from transformers import pipeline

classifier = pipeline("text-classification",
                      model="j-hartmann/emotion-english-distilroberta-base",
                      top_k=None,
                      device=1)

result = classifier("Some hype hip hop songs")
print(result)
