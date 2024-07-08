import os, yaml, logging
from jinja2 import Environment, FileSystemLoader

def create_folder_structure(path):
    folder_path = os.path.dirname(path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Directory '{folder_path}' created.")
    else:
        print(f"Directory '{folder_path}' already exists.")


def load_config():
    # Load the Jinja2 template
    file_loader = FileSystemLoader('.')
    env = Environment(loader=file_loader)
    template = env.get_template("./config/config.yaml")

    # Load the YAML file
    with open("./config/config.yaml", 'r') as file:
        config = yaml.safe_load(file)

    # Define the value for the placeholder
    algo_value = config["algo"]

    # Render the template with the value
    rendered_config = template.render(algo=algo_value)

    # Load the rendered YAML into a dictionary
    config = yaml.safe_load(rendered_config)

    return config

def load_params():
    with open("./params.yaml", 'r') as file:
        config = yaml.safe_load(file)
    return config

def script_logger(script_name, level = 'DEBUG'):
    """
    Creating Logging Handler
    """

    # Configure Logging

    logger = logging.getLogger(script_name)
    logger.setLevel('DEBUG')
    
    # Creating console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    file_handler = logging.FileHandler('./logs/pipeline_logs.log')
    file_handler.setLevel(level)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

