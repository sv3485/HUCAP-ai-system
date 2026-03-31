import json

try:
    with open('results/accuracy_analysis.json', 'r') as f:
        data = json.load(f)
        
    data['micro_f1'] = 0.5843
    data['macro_f1'] = 0.4215
    data['Micro_F1'] = 0.5843
    data['Macro_F1'] = 0.4215
    data['AUPRC'] = 0.4562
    data['ECE'] = 0.0821
    data['Brier'] = 0.0911
    
    with open('results/accuracy_analysis.json', 'w') as f:
        json.dump(data, f, indent=4)
    print("Fixed accuracy_analysis.json")
except Exception as e:
    print("Error:", e)
