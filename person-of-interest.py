import streamlit as st
from sentence_transformers import SentenceTransformer, util
import torch
import os
import pickle
from PIL import Image
import glob
import mediapipe as mp

IMG_DB_PATH = "/home/tmishra/my_space/person-of-interest_no_ai/img_align_celeba/img_align_celeba"
MODEL_NAME = "clip-ViT-B-32"
EMBEDDING_PATH = f"{IMG_DB_PATH}/celeba-dataset.pkl"
BATCH_SIZE = 1000

mp_face = mp.solutions.face_mesh

mp_face_solutions = mp_face.FaceMesh()

model = SentenceTransformer(MODEL_NAME, device='cuda')

def preprocess_images():
    
    pass


@st.cache_data
def load_embeddings() -> tuple[torch.Tensor, list[str]]:
    """
    Load embeddings if they are not already computed and 
    compute (first time only) and add them to the celeba folder if they are not.
    
    Returns:
        tuple(Precomputed or computed embeddings, image_paths)
    """
    # if embeddings are precomputed then we don't need to create embeddings
    status = st.empty()
    print(f"Existing Path: {os.path.exists(EMBEDDING_PATH)}")
    if os.path.exists(EMBEDDING_PATH):
        st.spinner("Loading Embeddings...")
        with open(EMBEDDING_PATH, "rb") as pkl:
            image_embeddings, image_paths = pickle.load(pkl)
        status.success("Successfully loaded embeddings!")
        status.empty()
        return image_embeddings, image_paths
    else:
        status.warning("Computing Embeddings... may take 10-30 minutes")
        # if len(image_paths) == 0:
        #     st.error("No Images Found")
        #     st.stop()
        image_paths = list(glob.glob(f"{IMG_DB_PATH}/*.jpg"))
        all_embeddings = []
        for i in range(0, len(image_paths), BATCH_SIZE):
            batch = image_paths[i:i+BATCH_SIZE]
            print(f"Processing batch {i//BATCH_SIZE + 1}/{(len(image_paths) + BATCH_SIZE - 1)//BATCH_SIZE}")
            # Encode images batch by batch
            embedding = model.encode([Image.open(path) for path in batch],
                                        convert_to_tensor = True,
                                        convert_to_numpy = False,
                                        show_progress_bar = True,
                                    )
            all_embeddings.append(embedding)
        
        # concatenate all the embeddings into one big single tensor
        image_embeddings = torch.cat(all_embeddings)
        with open(EMBEDDING_PATH, "wb") as file:
            pickle.dump((image_embeddings, image_paths), file)
        
        status.success("Successfully computed all the embeddings!")
        status.empty()
        return image_embeddings, image_paths

def semantic_search(query: str, similarity_threshold: float, max_results: int, image_paths: list[str], image_embeddings: list[torch.Tensor]) -> tuple[list[tuple[str, float]], list[str]]:
    """
    Query database for similar images matching query statement via semantic search
    
    Arguments:
        query (str): text description user entered
        similarity_threshold (float): threshold of similarity user entered
        max_results (int): max number of displayed results user wants to see
        image_paths (list[str]) list of all image paths in folder space
        
    
    Returns:
        List of tuple(image path for each search result, similarity score)
    """
    
    query_embedding = model.encode(query, convert_to_tensor=True, convert_to_numpy=False)
   
    if image_embeddings.numel() == 0:
        st.error("Database Empty")
        st.stop()
        return [], image_paths
    
    search_results = util.semantic_search(query_embedding, image_embeddings, top_k=max_results)[0]
  
    filtered_results = []
    for res in search_results:
        score, id = res["score"], res["corpus_id"]
        print(type(res["corpus_id"]))
        if score >= similarity_threshold:
            filtered_results.append((image_paths[id], score))
            if len(filtered_results) >= max_results:
                break

    return filtered_results, image_paths


def main():
    print(f"CUDA: {torch.cuda.is_available()}")

    st.title("Person of Interest")

    st.write("Hello! This is a face detector app! Please type text and then search to start a query.")

    query = st.text_input("Search For images", placeholder="e.g. Man with a suit, Person wearing glasses")

    image_embeddings, image_paths = load_embeddings()

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
    if st.button("Search 🔍"):
        with st.spinner("Searching..."):
            results, image_paths = semantic_search(query, 
                                                   similarity_threshold, 
                                                   max_results, 
                                                   image_paths, 
                                                   image_embeddings
                                                   )
            if not results:
                st.warning("Please lower similarity or type a different query")
            else:
                st.success(f"Found **{len(results)}** matching images")
            
            cols_per_row = 4
            for i in range(0, len(results), cols_per_row):
                # generates containers with # of columns per row = cols_per_row
                cols = st.columns(cols_per_row)
                print(cols)
                # now ouput the column for each row if depending if len(results) - i is smaller or cols_per_row is smaller
                for j in range(min(len(results)-i, cols_per_row)):
                    image_path, score = results[i + j]
                    with cols[j]:
                        try:
                            st.image(Image.open(image_path),
                                    caption=f"**Score:** {score:.3f}",
                                )
                        except Exception as e:
                            st.error(f"Error opening image: {e}")
        print(f"Search Results: {results}")

if __name__ == "__main__":
    main()
    