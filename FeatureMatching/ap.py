import Evaluation_utils as Eval
from pathlib import Path
import pandas as pd

'''
file_path = "/home/andrea/Desktop/Thesis_project/evaluation/000013/results_000053.jsonl"
Eval.plot_ap_curve_single_class(file_path,mode="rotation")
Eval.plot_ap_curve_single_class(file_path,mode="translation")
results = Eval.load_jsonl(file_path)
add_percentage,add_s_percentage = Eval.compute_add_percentage(results)
print("add   %",add_percentage)
print("add-s %",add_s_percentage)
'''
Eval.plot_all_jsonl_curves("/home/andrea/Desktop/Thesis_project/folders/48", mode="rotation")
Eval.plot_all_jsonl_curves("/home/andrea/Desktop/Thesis_project/folders/48", mode="translation")



'''
# Percorso base
base_dir = Path("/home/andrea/Desktop/Thesis_project/evaluation")

# Dati raccolti in un dizionario: {cartella: {file: (add, add-s)}}
data = {}

# Itera sulle cartelle da 000001 a 000020
for i in range(1, 21):
    folder_name = f"{i:06d}"
    folder_path = base_dir / folder_name

    if not folder_path.is_dir():
        print(f"Cartella non trovata: {folder_path}")
        continue

    print(f"\nProcessing in folder: {folder_name}")
    folder_results = {}

    for file in folder_path.glob("*.jsonl"):
        try:
            results = Eval.load_jsonl(str(file))
            add_percentage, add_s_percentage = Eval.compute_add_percentage(results)
            
            folder_results[file.name.replace("results_","").removesuffix(".jsonl")] = (add_percentage, add_s_percentage)
            print(f"[{folder_name}/{file.name}] add %: {add_percentage}, add-s %: {add_s_percentage}")
        except Exception as e:
            print(f"Errore nel file {file}: {e}")
            continue

    data[folder_name] = folder_results

# Creazione DataFrame pandas
df = pd.DataFrame.from_dict(data, orient="index")
df.index.name = "obj"
df.columns.name = "File"
df = df[sorted((df.columns))]
df = df.sort_index()
df = df.fillna("")

# Visualizza la tabella
#print("\nTabella finale:")
#print(df)

# (Opzionale) Salva su file CSV
df.to_html("tabella_risultati.html")

'''
