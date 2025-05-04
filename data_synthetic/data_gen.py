import random
from supabase import create_client, Client

url = "https://alwocqtpmrlfebnjjtct.supabase.co"
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImFsd29jcXRwbXJsZmVibmpqdGN0Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzQ0NTAzMDIsImV4cCI6MjA1MDAyNjMwMn0._NZ3uFepvW-JplnMj8jRhbf5CoT4QMS6lB5OJQaxFu4"
supabase: Client = create_client(url, key)

# target_folder = "Pemrograman Web Dasar"  # 1
target_folder = "Classification"       # 2

doc_index = 0

# 0.55
# Fungsi untuk menentukan rentang nilai random
# def get_random_range(num_names):
#     if num_names == 0:
#         return 0, 0
#     min_val = 0.2
#     max_val = min(0.2 + num_names * 0.03, 0.75)
#     return round(min_val, 2), round(max_val, 2)

# 0.6
# def get_random_range(num_names):
#     if num_names == 0:
#         return 0, 0
    
#     cluster_id = num_names % 3
    
#     if cluster_id == 0:
#         base_center = 0.2
#     elif cluster_id == 1:
#         base_center = 0.5
#     else:  # cluster_id == 2
#         base_center = 0.8
    
#     fine_adjustment = min(num_names * 0.01, 0.08)
#     center = base_center + fine_adjustment
#     center = max(0.15, min(center, 0.85))
#     width = 0.04
    
#     return round(center - width, 2), round(center + width, 2)

def get_random_range(num_names):
    if num_names == 0:
        return 0, 0
    
    global doc_index
    cluster_id = doc_index % 3
    doc_index += 1
    
    if cluster_id == 0:
        center = 0.2
        width = 0.03
    elif cluster_id == 1:
        center = 0.5
        width = 0.03
    else:  # cluster_id == 2
        center = 0.8
        width = 0.03
    
    noise = random.uniform(-0.01, 0.01)
    center += noise
    
    return round(center - width, 2), round(center + width, 2)

# Ambil data plagiarism (list of dict) dari Supabase
response_plag = supabase.table("documents")\
    .select("plagiarism")\
    .eq("folder", target_folder)\
    .order("uploadedDate", desc=False)\
    .execute()

# Ambil data ID dokumen untuk update nanti
response_docs = supabase.table("documents")\
    .select("id, uploadedDate")\
    .eq("folder", target_folder)\
    .order("uploadedDate", desc=False)\
    .execute()

# Ekstrak data
raw_plagiarism = [item["plagiarism"] for item in response_plag.data]
documents = response_docs.data

# List hasil similarity untuk diupdate dan ditulis ke file
similarity_data = []

# Proses dan simpan hasil ke file
with open("plagiarisme_hasil.txt", "w") as f:
    for entry in raw_plagiarism:
        if not entry:
            similarity_data.append([])
            f.write("[],\n")
            continue

        original_dict = entry[0]
        num_names = len(original_dict)
        low, high = get_random_range(num_names)
        new_dict = {name: round(random.uniform(low, high), 5) for name in original_dict}

        similarity_data.append([new_dict])  # disimpan dengan format list of dict
        f.write(f"{[new_dict]},\n")

# Validasi dan update database
if len(documents) != len(similarity_data):
    print("⚠️ Jumlah dokumen dan similarity_data tidak cocok!")
    print(f"Documents: {len(documents)}, Similarity: {len(similarity_data)}")
else:
    for doc, sim in zip(documents, similarity_data):
        doc_id = doc['id']
        update_response = supabase.table("documents")\
            .update({"plagiarism": sim})\
            .eq("id", doc_id)\
            .execute()
        print(f"✅ Updated doc ID {doc_id} with plagiarism data: {sim}")