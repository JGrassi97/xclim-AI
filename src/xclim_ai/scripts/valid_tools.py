from xclim_ai import TOOLS
from xclim_ai.utils.paths import VALID_TOOLS_PATH
from xclim_ai.datasets import load_dataset_from_config

import warnings
import yaml

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────

def get_valid_tools(ds):

    tools = [cls(ds=ds) for cls in TOOLS.values()]  
    variables = list(ds.data_vars.keys()) + ['lat']

    valid_tools = []

    # STEP 1: Filter tools based on required DataArray arguments
    for tool in tools:
        required = [
            arg for arg in tool.args_schema.model_fields.keys()
            if 'DataArray' in str(tool.args_schema.model_fields[arg])
        ]

        if all(arg in variables for arg in required):
            valid_tools.append(tool)

        else:
            continue

    print(f"Found {len(valid_tools)} valid tools in {len(tools)} tools.")

    # STEP 2: Check if the tool can be run with the given DataArray arguments
    rejected_tools = []
    for tool in valid_tools:
        try:
            #print(f"Running tool {tool.name} ")
            tool._run(safe=False)
        except Exception as e:
            rejected_tools.append(tool)
            continue

    valid_tools = [tool for tool in valid_tools if tool not in rejected_tools]
    print(f"Found {len(valid_tools)} valid tools after running them.")

    return valid_tools

# ─────────────────────────────────────────────────────────────────────────────

def main():

    # Load dataset using configuration
    ds = load_dataset_from_config(start_date="1980-01-01")

    valid_tools = get_valid_tools(ds)

    with open(VALID_TOOLS_PATH, "w") as f:
        yaml.dump([tool.name for tool in valid_tools], f, default_flow_style=False)

    print(f"✅ {len(valid_tools)} valid tools saved.")