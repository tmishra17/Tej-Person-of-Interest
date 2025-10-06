import streamlit as st
from sentence_transformers import SentenceTransformer, util
import torch
import os
import pickle
from PIL import Image
import glob

IMG_DB_PATH = "/home/tmishra/my_space/person-of-interest_no_ai/img_align_celeba/img_align_celeba"
MODEL_NAME = "clip-ViT-B-32"
EMBEDDING_PATH = f"/home/tmishra/my_space/person-of-interest_no_ai/celeba-dataset.pkl"
BATCH_SIZE = 1000

model = SentenceTransformer(MODEL_NAME, device='cuda')



def load_embeddings():
    """
    Load embeddings if they are not already computed and 
    compute (first time only) and add them to the celeba folder if they are not.
    
    Returns:
        -Precomputed or computed embeddings
    """
    # if embeddings are precomputed then we don't need to create embeddings
    status=st.empty()
    if os.path.exists(EMBEDDING_PATH):
        status.info("Loading Embeddings...")
        status.spin()
        with open(EMBEDDING_PATH, "rb") as pkl:
            pkl_file=pickle.load(pkl)
            
        print(type(pkl_file))
        
        image_embeddings = model.encode([Image.open(image) for image in pkl_file],
                                        convert_to_tensor = True,
                                        convert_to_numpy = False,
                                        batch_size=BATCH_SIZE
                                    )
        
        status.success("Successfully loaded embeddings")
        status.remove()
        return image_embeddings
    else:
        status.info("Computing Embeddings... may take 10-30 minutes")
        image_paths = glob.glob(f"{IMG_DB_PATH}/*.jpg")
        if len(image_paths) == 0:
            st.error("No Images Found")
            st.stop()
        image_embeddings = []
        for i in range(10):
            embedding = model.encode(image_paths[i],
                                     convert_to_tensor = True,
                                     conver_to_numpy = False,
                                     )
            image_embeddings.append((image_paths, embedding))
        
        with open(EMBEDDING_PATH, "wb"):
            pickle.dump(image_embeddings)
        
        
        return image_embeddings

def semantic_search(query: str, similarity_threshold: float, max_results:int ):
    image_embeddings = load_embeddings()
    search_results = util.semantic_search(query, image_embeddings)
    i = 0
    res_count = 0
    filtered_results = []
    while i < len(search_results) and res_count < max_results:
        if search_results[i] >= similarity_threshold:
            filtered_results.append((search_results[i], search_results[i].score))
    
    return filtered_results


def main():
    print(torch.cuda.is_available())

    st.title("Person of Interest")

    st.write("Hello! This is a face detector app! Please type text and then search to start a query.")

    query = st.text_input("Search For images", placeholder="e.g. Man with a suit, Person wearing glasses")

    

    st.sidebar.header("⚙️ Search Settings")
    
    with st.sidebar:
        similarity_threshold=st.slider("Similarity Threshold",
                                    min_value=0.0,
                                    max_value=0.5,
                                    value=0.2,
                                    step=.01,
                                    help="Select Similarity to Query"
                                )
        
        max_results=st.slider("Max Results",
                            min_value=1,
                            max_value=50,
                            value=10,
                            step=1,
                            help="Select Number of results"
                        )
    st.button("Search 🔍")

    search_results = semantic_search(query, similarity_threshold, max_results)
    # for sr in search_results:
    #     st.image()
    
if __name__ == "__main__":
    main()
    