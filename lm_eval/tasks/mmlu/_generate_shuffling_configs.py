"""
Take in a YAML, and output all "other" splits with this YAML
"""
import os
import yaml
import shutil

from tqdm import tqdm

from lm_eval import utils

SUBJECTS = {
    "abstract_algebra": "stem",
    "anatomy": "stem",
    "astronomy": "stem",
    "business_ethics": "other",
    "clinical_knowledge": "other",
    "college_biology": "stem",
    "college_chemistry": "stem",
    "college_computer_science": "stem",
    "college_mathematics": "stem",
    "college_medicine": "other",
    "college_physics": "stem",
    "computer_security": "stem",
    "conceptual_physics": "stem",
    "econometrics": "social_sciences",
    "electrical_engineering": "stem",
    "elementary_mathematics": "stem",
    "formal_logic": "humanities",
    "global_facts": "other",
    "high_school_biology": "stem",
    "high_school_chemistry": "stem",
    "high_school_computer_science": "stem",
    "high_school_european_history": "humanities",
    "high_school_geography": "social_sciences",
    "high_school_government_and_politics": "social_sciences",
    "high_school_macroeconomics": "social_sciences",
    "high_school_mathematics": "stem",
    "high_school_microeconomics": "social_sciences",
    "high_school_physics": "stem",
    "high_school_psychology": "social_sciences",
    "high_school_statistics": "stem",
    "high_school_us_history": "humanities",
    "high_school_world_history": "humanities",
    "human_aging": "other",
    "human_sexuality": "social_sciences",
    "international_law": "humanities",
    "jurisprudence": "humanities",
    "logical_fallacies": "humanities",
    "machine_learning": "stem",
    "management": "other",
    "marketing": "other",
    "medical_genetics": "other",
    "miscellaneous": "other",
    "moral_disputes": "humanities",
    "moral_scenarios": "humanities",
    "nutrition": "other",
    "philosophy": "humanities",
    "prehistory": "humanities",
    "professional_accounting": "other",
    "professional_law": "humanities",
    "professional_medicine": "other",
    "professional_psychology": "social_sciences",
    "public_relations": "social_sciences",
    "security_studies": "social_sciences",
    "sociology": "social_sciences",
    "us_foreign_policy": "social_sciences",
    "virology": "other",
    "world_religions": "humanities",
}


def generate_configs(base_yaml_path, save_prefix_path, task_prefix, group_prefix):
    # get filename of base_yaml so we can `"include": ` it in our "other" YAMLs.
    base_yaml_name = os.path.split(base_yaml_path)[-1]
    
    ALL_CATEGORIES = []
    for subject, category in tqdm(SUBJECTS.items()):

        if category not in ALL_CATEGORIES:
            ALL_CATEGORIES.append(category)
 
        description = f"The following are multiple choice questions (with answers) about {' '.join(subject.split('_'))}.\n\n"

        yaml_dict = {
            "include": base_yaml_name,
            "group": f"mmlu_{task_prefix}_{category}"
            if task_prefix != ""
            else f"mmlu_{category}",
            "group_alias": category.replace("_", " "),
            "task": f"mmlu_{task_prefix}_{subject}"
            if task_prefix != ""
            else f"mmlu_{subject}",
            "task_alias": subject.replace("_", " "),
            "dataset_name": subject,
            "description": description,
        }

        file_save_path = f"mmlu_{subject}.yaml"
        # eval_logger.info(f"Saving yaml for subset {subject} to {file_save_path}")
        with open(file_save_path, "w") as yaml_file:
            yaml.dump(
                yaml_dict,
                yaml_file,
                # width=float("inf"),
                allow_unicode=True,
                default_style='"',
            )

    if task_prefix != "":
        mmlu_subcategories = [
            f"mmlu_{task_prefix}_{category}" for category in ALL_CATEGORIES
        ]
    else:
        mmlu_subcategories = [f"mmlu_{category}" for category in ALL_CATEGORIES]

    if group_prefix != "":
        file_save_path = group_prefix + ".yaml"
    else:
        file_save_path = save_prefix_path + ".yaml"
    
    file_save_path = "_mmlu.yaml"

    with open(file_save_path, "w") as yaml_file:
        yaml.dump(
            {
                "group": f"mmlu_{task_prefix}"
                if task_prefix != ""
                else "mmlu",
                "task": mmlu_subcategories,
            },
            yaml_file,
            indent=4,
            default_flow_style=False,
        )

PROCESS_DOC_FSTRING = '''
import random

def _process_doc(doc):
    def format_example(doc, keys):
        """
        Question: <prompt>
        Choices:
        A. <choice1>
        B. <choice2>
        C. <choice3>
        D. <choice4>
        Answer:
        """
        prompt = "Question: " + doc["question"] + "\\nChoices:\\n"
        prompt += "".join(
            [f"{key}. {choice}\\n" for key, choice in sorted(zip(keys, doc["choices"]), key=lambda _: random.random())]
        )
        prompt += "Answer:"
        return prompt
    
    keys = ["A", "B", "C", "D"]
    return {
        "question": format_example(doc, keys),
        "choices": keys,
        "answer": doc["answer"],
    }

def process_docs(dataset):
    return dataset.map(_process_doc)
'''

if __name__ == "__main__":
    exp_name = f'shuffle_options'
    os.makedirs(exp_name, exist_ok=True)
    os.chdir(exp_name)
    base_yaml_path = '../_default_selection_bias_template_yaml'
    shutil.copyfile(base_yaml_path, '_default_template_yaml')

    with open('utils.py', 'w') as f:
        f.write(PROCESS_DOC_FSTRING)

    generate_configs(base_yaml_path='_default_template_yaml', save_prefix_path=exp_name, task_prefix=exp_name, group_prefix=f'{exp_name}_mmlu')
