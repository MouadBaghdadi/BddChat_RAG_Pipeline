import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM

class DahakChatPipeline:
    def __init__(self, model_id="google/gemma-2b-it", device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model_id = model_id
        self.device = device
        self.embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2", device=device)
        self.llm_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_id, torch_dtype=torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_id)
        self.context_data = pd.read_csv('bdd.csv')
        self.pages_and_chunks = self.context_data.to_dict(orient="records")
        self.embeddings = torch.tensor(np.stack(self.context_data['embeddings'].apply(lambda x: np.fromstring(x.strip("[]"), sep=', ')), axis=0))

    def ask_dahak(self, query, temperature=0.7, max_new_token=256):
        context = self.top_scores_indexs(query=query, top_n=5)
        prompt = self.prompt_augment(query, context)
        inputs = self.tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
        outputs = self.llm_model.generate(input_ids=inputs, temperature=temperature, max_new_tokens=max_new_token)
        output_text = self.tokenizer.decode(outputs[0])
        output_text = output_text.replace(prompt," ").replace("<bos>", " ").replace("<eos>"," ")
        return output_text

    def top_scores_indexs(self, query, top_n):
        query_embedded = self.embedding_model.encode(query, convert_to_tensor=True)
        dot_scores = util.dot_score(a=query_embedded.float(), b=self.embeddings.float())[0]
        top_scores_indexs = torch.topk(dot_scores, k=top_n)
        return_list = []
        for value, index in zip(top_scores_indexs.values.tolist(), top_scores_indexs.indices.tolist()):
            values_map = {
                "score": value,
                "text": self.pages_and_chunks[index]['sentence_chunk'],
                "page": self.pages_and_chunks[index]['page_number']
            }
            return_list.append(values_map)
        return return_list

    def prompt_augment(self, query, context_items):
        context = "- " + "\n- ".join([item['text'] for item in context_items])
        base_prompt = f"""if the question isn't in the field and the scope of database or there is no context items attached with the question then don't answer it, otherwise based on these context items, please answer the question:
    context_items: {context}
    query: {query}
    Answer: 
    """
        dialogue_template = [
            {"role": "user",
             "content": base_prompt}
        ]
        prompt = self.tokenizer.apply_chat_template(conversation=dialogue_template, tokenize=False, add_generation_prompt=True)
        return prompt











#         import torch
# import numpy as np
# import pandas as pd
# from sentence_transformers import SentenceTransformer, util

# class DahakChatPipeline:
#     def __init__(self):
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2", device=self.device)
#         self.context_data = pd.read_csv('bdd.csv')
#         self.pages_and_chunks = self.context_data.to_dict(orient="records")
#         self.embeddings = torch.tensor(np.stack(self.context_data['embeddings'].apply(lambda x: np.fromstring(x.strip("[]"), sep=', ')), axis=0))

#     def ingest_pdf(self, pdf_path):
#         # Placeholder for PDF ingestion logic
#         pass

#     def start(self):
#         # Placeholder for pipeline initialization logic
#         pass

#     def query(self, query):
#         top_n = 5
#         query_embedding = self.embedding_model.encode(query, convert_to_tensor=True).to(self.device)
#         dot_scores = util.dot_score(a=query_embedding.float(), b=self.embeddings.float())[0]
#         top_values = torch.topk(dot_scores, k=top_n)

#         responses = []
#         for value, index in zip(top_values.values.tolist(), top_values.indices.tolist()):
#             response = {
#                 "score": value,
#                 "text": self.pages_and_chunks[index]['sentence_chunk'],
#                 "page": self.pages_and_chunks[index]['page_number']
#             }
#             responses.append(response)

#         return responses