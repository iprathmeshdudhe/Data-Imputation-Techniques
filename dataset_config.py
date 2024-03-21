class HopperConfig:
    def __init__(self):
        self.data_name = "Hopper-v4"
        self.state_vector = ["obs_1", "obs_2", "obs_3", "obs_4", "obs_5", 
                             "obs_6", "obs_7", "obs_8", "obs_9", "obs_10", 
                             "obs_11"]
        self.missing_state_vector = ["obs_6", "obs_7"]
        self.action_vector = []
        self.full_data_path = "dataset/Hopper-v4.csv"
        self.missing_data_path = "dataset/Hopper-v4_w_missing_values_random.csv"
        self.imputed_data_path = "Imputation_results/hopper_imputed"

    
class MimicConfig:
    def __init__(self):
        self.data_name = "mimiciv_2"
        '''self.state_vector = ["vent_etco2", "blood_paco2", "blood_pao2", "vent_fio2", 
                             "vital_spo2", "vital_hr", "vent_rrtot"]
        self.missing_state_vector = ["blood_paco2", "blood_pao2"]'''

        self.state_vector = ["vent_vtnorm","vent_mairpress","blood_be","cum_fluid_balance","blood_sodium",
                             "vital_SBP","state_ivfluid4h","vent_mv","blood_hco3","vent_vt","vent_inspexp",
                             "blood_paco2","vent_rrtot","blood_chlorid","blood_hb","vent_mode","vital_map","blood_hct",
                             "vent_peep","vital_DBP","blood_ph","vent_pinsp","vent_suppress"]
        self.missing_state_vector = ["blood_be", "vital_map", "vent_pinsp"]
        self.action_vector = []
        self.full_data_path = "dataset\\240126_no_impute\\mimiciv_state_vectors_40_corr.csv"
        self.missing_data_path = "dataset/mimiciv_2_w_missing_values_random.csv"
        self.imputed_data_path = "Imputation_results/mimiciv_2_imputed"

    
class TudConfig:
    def __init__(self):
        self.data_name = "tud"
        self.state_vector = ["SaO2","BEa","pHa","Lactat","SpO2","HaCO3","PaCO2"]
        self.missing_state_vector = ["BEa", "pHa"]
        self.action_vector = []
        self.full_data_path = "dataset\\dresden_preprocess_benchmark_checkpoints\\tud_final_state_vectors_20.csv"
        self.missing_data_path = "dataset/tud_w_missing_values_random.csv"
        self.imputed_data_path = "Imputation_results/tud_imputed"
