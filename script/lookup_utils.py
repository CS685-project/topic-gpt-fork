import json

# TODO: Modify this file if you require a different lookup strategy
#       (e.g., API calls instead of hard-coded JSON file)

DEPLOYMENT_LOOKUP = json.load(open("../lookups/deployment_lookup.json"))


def check_if_in_lookup_file(deployment_name, provider):
    if provider not in DEPLOYMENT_LOOKUP:
        raise KeyError(f"{provider} is not present in the lookup file.")
    if deployment_name not in DEPLOYMENT_LOOKUP[provider]:
        raise KeyError(f"The lookup file does not contain model {deployment_name} for provider {provider}.")
    if "price" not in DEPLOYMENT_LOOKUP[provider][deployment_name]:
        raise KeyError(f"Please add a 'price' to model {deployment_name} of provider {provider}.")


def get_context_length(deployment_name, provider):
    check_if_in_lookup_file(deployment_name, provider)
    return DEPLOYMENT_LOOKUP[provider][deployment_name]["context"]


def get_api_string(deployment_name, provider):
    check_if_in_lookup_file(deployment_name, provider)
    return DEPLOYMENT_LOOKUP[provider][deployment_name]["api_string"]


def get_price(deployment_name, provider):
    check_if_in_lookup_file(deployment_name, provider)
    return DEPLOYMENT_LOOKUP[provider][deployment_name]["price"]
