import requests
import json
import numpy as np
import pickle
import config
from sklearn.preprocessing import StandardScaler as ss


URL = config.URL
headers = config.HEADERS

def protein_carb_ratio(protein, carb):
    return protein / (carb + .1)

def sugar_sodium_fats_over_fiber(sugar, sodium, fats, fiber):
    return (sugar + sodium + fats) / (fiber + .1)

def fibers_saturated_fat_ratio(fiber,saturated_fat):
    return fiber / (saturated_fat + .1)

def energy(carbs, proteins, fat):
    return 4.189 * ((carbs * 4) + (proteins * 5)+(fat * 9))

loaded_model = pickle.load(open("random_forest_1.sav", 'rb'))
food = input("Please enter a food item: ")
while (food != "quit"):
    try:
        data = {'query':food}
        #print(data)
        r = requests.post(URL, headers=headers, data=json.dumps(data))
        nutrients = r.json()
        nutrientsPP = (json.dumps(r.json(), indent=4))
        #print(nutrientsPP)
        values = nutrients["foods"][0]['full_nutrients']

        flattened = {k['attr_id']:k['value'] for k in values}


        food_api_map = {'energy_kj': 268, 
                        'fats': 204, 
                        'saturated_fat_g': 606, 
                        'carbs_g': 205, 
                        'sugars_g': 269, 
                        'fiber_g': 291, 
                        'proteins_g': 203, 
                        'sodium_mg': 307}

        scaler = ss()

        npa = np.array([flattened[268]
                        ,flattened[204]*100
                        ,flattened[606]*100
                        ,flattened[205]*100
                        ,flattened[269]*100
                        ,flattened[291]*100
                        ,flattened[203]*100
                        ,(flattened[307]/1000)*100
                        ,protein_carb_ratio(flattened[203]*100,flattened[205]*100)
                        ,sugar_sodium_fats_over_fiber(flattened[269]*100,(flattened[307]/1000)*100,flattened[204]*100,flattened[291]*100)
                        ,fibers_saturated_fat_ratio(flattened[291]*100,flattened[606]*100)])
        
        X = scaler.fit_transform(npa[:,np.newaxis]).reshape(1,-1)
        
        print("\nRetrieving Nutritional Data for: {}".format(data["query"].upper()))
        print("Serving Size: {} {}".format(nutrients["foods"][0]["serving_qty"], nutrients["foods"][0]["serving_unit"]))
        print("Energy (Kj): {}".format(flattened[268]))
        print("Fats (g): {}".format(flattened[204]))
        print("Saturated Fats (g): {}".format(flattened[606]))
        print("Carbohydrate (g): {}".format(flattened[205]))
        print("Sugar (g): {}".format(flattened[269]))
        print("Fiber (g): {}".format(flattened[291]))
        print("Protein (g): {}".format(flattened[203]))
        print("Sodium (mg): {}".format(flattened[307]))

        print("French Nutritional Grade (A - E): {}".format(loaded_model.predict(X)[0].upper()))
    except KeyError:
        print("Missing feature values to calculate result.")
    finally:
        food = input("Please enter a food item: ")