import Evaluation_utils as Eval

file_path = "/home/andrea/Desktop/Thesis_project/results_000049.jsonl"
Eval.plot_ap_curve_single_class(file_path,mode="rotation")
Eval.plot_ap_curve_single_class(file_path,mode="translation")
results = Eval.load_jsonl(file_path)
add_percentage,add_s_percentage = Eval.compute_add_percentage(results)
print("add   %",add_percentage)
print("add-s %",add_s_percentage)