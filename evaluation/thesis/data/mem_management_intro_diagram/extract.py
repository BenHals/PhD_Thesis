import pickle 
import json
with open("ff_res_AU5.pickle", 'rb') as f:
    data = pickle.load(f)
    
print(list(data.keys()))
for k in data:
    print(k)
    v = data[k]
    print(type(v))
    print(v)

new_data = {}
new_data['r'] = data['r']
new_data['z'] = data['z'].tolist()

with open("ff_res_AU5.json", 'w') as f:
    json.dump(new_data, f)
with open("ff_res_BU5.pickle", 'rb') as f:
    data = pickle.load(f)
    
print(list(data.keys()))
for k in data:
    print(k)
    v = data[k]
    print(type(v))
    print(v)

new_data = {}
new_data['r'] = data['r']
new_data['z'] = data['z'].tolist()

with open("ff_res_BU5.json", 'w') as f:
    json.dump(new_data, f)