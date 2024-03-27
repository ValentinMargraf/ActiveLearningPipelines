from py_experimenter.experimenter import PyExperimenter

exp_config_file = "config/exp_conf.yml"
db_config_file = "config/db_conf.yml"

experimenter = PyExperimenter(experiment_configuration_file_path=exp_config_file,
                              database_credential_file_path=db_config_file)
