import faiss

index_path = r"C:\Users\neeraj_sachdeva\Desktop\Final_Demo\Demo\Scripts\faiss_indices\video.index"
index = faiss.read_index(index_path)
print("Number of vectors:", index.ntotal)
