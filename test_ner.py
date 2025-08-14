from ner.model.pipeline_ner import ner_pipeline


prompt = "chill song like moon by kanye"
output = ner_pipeline(prompt)
print(output)