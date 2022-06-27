import os
import requests
import json
#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000"



#Call each API endpoint and store the responses
response1 = requests.post(os.path.join(URL,"prediction"),json={"filepath": "testdata/testdata.csv"}).text#put an API call here

response2 = requests.get(os.path.join(URL,"scoring")).text#put an API call here

response3 = requests.get(os.path.join(URL,"summarystats")).text#put an API call here
response4 = requests.get(os.path.join(URL,"diagnostics")).text #put an API call here

print(response1)


print(response2)
print(response3)
print(response4)



#combine all API responses
#responses = #combine reponses here
responses = str(response1) + "\n" + str(response2) + "\n" + str(response3) + "\n" + str(response4)


#write the responses to your workspace

with open('config.json', 'r') as file:
    config = json.load(file)
    model_path = os.path.join(config['output_model_path'])

with open(os.path.join(model_path, "apireturns.txt"), "w") as file:
    file.write(responses)

