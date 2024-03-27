from py_experimenter.experimenter import PyExperimenter, ResultProcessor

exp_config_file = "config/exp_conf.yml"
db_config_file = "config/db_conf.yml"
setup_table = False


def run_experiment(parameters: dict, result_processor: ResultProcessor, custom_config: dict):
    pass


experimenter = PyExperimenter(experiment_configuration_file_path=exp_config_file,
                              database_credential_file_path=db_config_file)

if setup_table:
    experimenter.fill_table_from_config()
