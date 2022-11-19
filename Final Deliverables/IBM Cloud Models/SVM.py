import requests

# NOTE: you must manually set API_KEY below using information retrieved from your IBM Cloud account.
API_KEY = "xU7wUCiLdUHS2Y9iDdLma2qNSXQyWBzohz-ZlQCOB1Ss"
token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey":
 API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
mltoken = token_response.json()["access_token"]

header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}

# NOTE: manually define and pass the array(s) of values to be scored in the next line
payload_scoring = {"input_data": [{"field": [['sensor2','sensor3','sensor4','sensor7','sensor8','sensor9','sensor15','sensor17','sensor20','sensor21']], "values": [[0.310241, 0.304556, 0.386226, 0.618357, 0.257576, 0.208068, 0.495575, 0.416667, 0.651163, 0.442833]]}]}
# payload_scoring = {"input_data": [{"field": [['sensor2','sensor3','sensor4','sensor7','sensor8','sensor9','sensor15','sensor17','sensor20','sensor21']], "values": [[0.183735,	0.406802,	0.309757,	0.726248,	0.242424,	0.109755,	0.363986,	0.333333	,0.713178,	0.724662]]}]}
print("Sent API call to model stored on IBM Cloud")

response_scoring = requests.post('https://us-south.ml.cloud.ibm.com/ml/v4/deployments/88b291c8-3850-4c4b-884c-2e617bad6870/predictions?version=2022-11-16', json=payload_scoring,
 headers={'Authorization': 'Bearer ' + mltoken})
print("Scoring response")
prediction = response_scoring.json()

print(prediction)