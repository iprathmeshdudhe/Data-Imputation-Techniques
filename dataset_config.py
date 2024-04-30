class MIMICIV_config:
    def __init__(self):
        self.data_name = 'MIMICIV'
        self.full_data_path = 'dataset/mimiciv_full_data.csv'
        self.imputed_data_path = 'dataset/mimiciv_blood_chlorid_imp'        # Yes, it is correct, don't add .csv extension
        self.missing_data_path = 'dataset/mimiciv_blood_chlorid_w_missing_values_random.csv'


        # These are only useful for GRU (nn) imputation and evaluation
        '''
        # Indepedent variables
        self.state_vector = []'''

        # Target variable
        self.missing_state_vector = ["blood_chlorid"]