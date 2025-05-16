import json
import os
root = os.path.dirname(__file__)
from collections import defaultdict
from copy import deepcopy

def get_config(company):
    """
    return configuration json files for ChatChain
    user can customize only parts of configuration json files, other files will be left for default
    Args:
        company: customized configuration name under CompanyConfig/

    Returns:
        path to three configuration jsons: [config_path, config_phase_path, config_role_path]
    """
    config_dir = os.path.join(root, "CompanyConfig", company)
    default_config_dir = os.path.join(root, "CompanyConfig", "Default")

    config_files = [
        "ChatChainConfig.json",
        "PhaseConfig.json",
        "RoleConfig.json"
    ]

    config_paths = []

    for config_file in config_files:
        company_config_path = os.path.join(config_dir, config_file)
        default_config_path = os.path.join(default_config_dir, config_file)

        if os.path.exists(company_config_path):
            config_paths.append(company_config_path)
        else:
            config_paths.append(default_config_path)

    return tuple(config_paths)


config_path, config_phase_path, config_role_path = get_config("Default")

with open(config_path, 'r', encoding="utf8") as file:
    config = json.load(file)  # chat chain需要变化cycleNum
with open(config_phase_path, 'r', encoding="utf8") as file:
    config_phase = json.load(file)  # phase需要变化模型选择
with open(config_role_path, 'r', encoding="utf8") as file:
    config_role = json.load(file)  # role不用变



cycle_number_choices = {
"CodeCompleteAll": [1, 10, 30],
"CodeReview": [1, 10, 30],
"Test": [1, 10, 30],
}

config_chatchain_list = []
config_phase_list = []


def assign_chatchain_cycleNum(n1, n2, n3):
    config_copy = deepcopy(config)
    for key, value in config_copy.items():
        config_copy[key] = value
        if key == 'chain':
            for e in config_copy['chain']:  # e = config_copy['chain'][3], 4,5
                if "cycleNum" in e:
                    phase = e['phase']
                    if phase == "CodeCompleteAll":
                        e['cycleNum'] = n1
                    elif phase == "CodeReview":
                        e['cycleNum'] = n2
                    elif phase == "Test":
                        e['cycleNum'] = n3
    return config_copy

for n1 in cycle_number_choices["CodeCompleteAll"]:
    for n2 in cycle_number_choices["CodeReview"]:
        for n3 in cycle_number_choices["Test"]:
            config_chatchain_list.append(assign_chatchain_cycleNum(n1, n2, n3))


model_choices = {
    'DemandAnalysis': 'llama70b',
    'LanguageChoose': 'llama70b',
    'Coding': ['llama70b', 'llama8b', 'llama3b'],
    "CodeComplete": ['llama70b', 'llama8b', 'llama3b'],
    'ArtDesign': 'llama70b',
    'ArtIntegration': 'llama70b',
    'CodeReviewComment': ['llama70b', 'llama8b', 'llama3b'],
    'CodeReviewModification': "",
    'TestErrorSummary': ['llama70b', 'llama8b', 'llama3b'],
    'TestModification': "",
    'EnvironmentDoc': 'llama70b',
    'Manual': 'llama70b',
}

def assign_phase_models(da_model, lc_model, coding_model, cc_model, ad_model, ai_model, crc_model, crm_model, tes_model, tm_model, ed_model, manual_model):
    config_phase_copy = deepcopy(config_phase)
    for key, value in config_phase_copy.items():
        if key == "DemandAnalysis":
            config_phase_copy[key]['model'] = da_model
        elif key == "LanguageChoose":
            config_phase_copy[key]['model'] = lc_model
        elif key == "Coding":
            config_phase_copy[key]['model'] = coding_model
        elif key == "CodeComplete":
            config_phase_copy[key]['model'] = cc_model
        elif key == "ArtDesign":
            config_phase_copy[key]['model'] = ad_model
        elif key == "ArtIntegration":
            config_phase_copy[key]['model'] = ai_model
        elif key == "CodeReviewComment":
            config_phase_copy[key]['model'] = crc_model
        elif key == "CodeReviewModification":
            config_phase_copy[key]['model'] = crm_model
        elif key == "TestErrorSummary":
            config_phase_copy[key]['model'] = tes_model
        elif key == "TestModification":
            config_phase_copy[key]['model'] = tm_model
        elif key == "EnvironmentDoc":
            config_phase_copy[key]['model'] = ed_model
        elif key == "Manual":
            config_phase_copy[key]['model'] = manual_model
        else:
            raise Exception("unknown key {}".format(key))
    return config_phase_copy

for cc_model in model_choices['CodeComplete']:
    for crc_model in model_choices['CodeReviewComment']:
        for tes_model in model_choices['TestErrorSummary']:
            config_phase_copy = assign_phase_models(
                model_choices['DemandAnalysis'],
                model_choices['LanguageChoose'],
                model_choices['Coding'],
                cc_model,
                model_choices['ArtDesign'],
                model_choices['ArtIntegration'],
                crc_model,
                crc_model,
                tes_model,
                tes_model,  # 改用一样的
                model_choices['EnvironmentDoc'],
                model_choices['Manual']
            )
            config_phase_list.append(config_phase_copy)

count = 0
print(len(config_chatchain_list)) # 27
print(len(config_phase_list)) # 27
for config_chatchain_cand in config_chatchain_list:
    for config_phase_cand in config_phase_list:
        config_dir = os.path.join(root, "CompanyConfig", f"test_{count+1}")
        os.makedirs(config_dir, exist_ok=True)
        with open(os.path.join(config_dir, "PhaseConfig.json"), 'w', encoding="utf8") as f:
            json.dump(config_phase_cand, f, indent=2)
        with open(os.path.join(config_dir, "ChatChainConfig.json"), 'w', encoding="utf8") as f:
            json.dump(config_chatchain_cand, f, indent=2)
        with open(os.path.join(config_dir, "RoleConfig.json"), 'w', encoding="utf8") as f:
            json.dump(config_role, f, indent=2)
        print(f'{count+1} done.')
        count += 1
