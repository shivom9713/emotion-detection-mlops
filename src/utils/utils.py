import os, yaml
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

