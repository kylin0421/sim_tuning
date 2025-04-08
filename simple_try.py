#venv name: sim_tuning
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import random
import re
from tqdm import tqdm


from sklearn.preprocessing import StandardScaler
import umap
import hdbscan
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt



def get_embedding(prompt,model,tokenizer,layer=-1):

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    last_hidden = outputs.hidden_states[layer]
    embedding = last_hidden.mean(dim=1)  # [batch_size, hidden_dim]
    return embedding.squeeze().cpu().numpy()

def get_natural_instructions(random_select=True,task_num=1):
    folder_path="/srv/data/sul/sim_tuning/natural_instructions"
    all_files = [f for f in os.listdir(folder_path) if f.endswith(".json")]

    if random_select:             
        selected_file = random.choice(all_files)
        selected_path = os.path.join(folder_path, selected_file)
            
    else:
        pattern=re.compile(rf"subtask0*{int(task_num)}_.+\.json")
        selected_file=None
        for file in all_files:
            if pattern.match(file):
                selected_file=file
        assert selected_file!=None,"No matching file found!"
        selected_path = os.path.join(folder_path, selected_file)

         
    print("selected file:",selected_path)

    with open(selected_path,"r") as f:
        task=json.load(f)

    print("Title:",task["Title"])
    print("Definition:",task["Definition"])

    prompts = [inst["input"] for inst in task["Instances"]]

    return prompts







def main(use_new_matrix=False):

    if use_new_matrix:
        model_name = "mistralai/Mistral-7B-Instruct-v0.1"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
        prompts1=get_natural_instructions(False,task_num=10)   #a list
        print("First length:",len(prompts1))
        prompts2=get_natural_instructions(False,task_num=3)
        print("Second length:",len(prompts2))
        embs1=[get_embedding(prompt,model,tokenizer,-1) for prompt in tqdm(prompts1,desc="Getting embeddings 1")]  #a list of np.ndarray(4096,)
        embs2=[get_embedding(prompt,model,tokenizer,-1) for prompt in tqdm(prompts2,desc="Getting embeddings 2")]
        embs_matrix=np.vstack(embs1+embs2)
        print(type(embs_matrix))
        print(embs_matrix.shape)
        assert embs_matrix.shape[0]==len(prompts1)+len(prompts2) and embs_matrix.shape[1]==4096, "embs_matrix shape not valid!"
        np.save("embs_matrix.npy",embs_matrix)
    
    
    X=np.load("embs_matrix.npy")
    print("X shape:",X.shape)   #(332+430,4096)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("X_scaled shape:",X_scaled.shape)


    reducer = umap.UMAP(n_components=50, random_state=42)
    X_reduced = reducer.fit_transform(X_scaled)
    print("X_reduced shape:",X_reduced.shape)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=300,min_samples=2)  #adjustable
    labels = clusterer.fit_predict(X_reduced)
    np.save("cluster_labels.npy", labels)
    print("Clustering result saved!")


    X_vis = PCA(n_components=2).fit_transform(X_reduced)
    print("X_vis shape:",X_vis.shape)
    plt.scatter(X_vis[:, 0], X_vis[:, 1], c=labels, cmap='Spectral', s=10)
    plt.title("HDBSCAN clustering result visualization")
    plt.colorbar()
    plt.show()






if __name__=="__main__":
    main()