import json
class MIMICIV_config:
    def __init__(self):

        f = open('imp_config.json')
        data_info = json.load(f)

        self.data_name = data_info["name"]
        self.full_data_path = data_info["full_data_path"]
        self.imputed_data_path = data_info["imputed_data_path"]
        self.missing_data_path = data_info["missing_data_path"]


        # These are only useful for GRU (nn) imputation and evaluation
        
        # Indepedent variables
        self.state_vector = data_info["state_vector"]

        # Target variable
        self.missing_state_vector = data_info["missing_state_vector"]


        
