from sentence_transformers import SentenceTransformer

class MyEmbeddings:
    def __init__(self, model_name="dunzhang/stella_en_1.5B_v5"):
        # Initialize the model with remote code trust
        self.model = SentenceTransformer(model_name, trust_remote_code=True,device="cuda")

    def __call__(self, text):
        # Ensure the text is a list for the model's encode method
        if isinstance(text, str):
            text = [text]
        return self.model.encode(text)

    def embed_documents(self, documents):
        # Call the encode method on the provided documents
        return self.model.encode(documents)

    def embed_query(self, query):
        # Call the encode method on the provided query
        l = self.model.encode(query)
        print(l)
    
        return l


embedder=MyEmbeddings()
# Define sentences to embed
sentences = "The weather is lovely today."


# Generate embeddings for the sentences
embeddings = embedder.embed_query(sentences)
print(embeddings)
