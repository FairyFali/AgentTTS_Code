import argparse
import json
import logging
import os
import shutil
import sys
import wandb
import time

from camel.typing import ModelType
from chatdev.eval_quality import main as eval

root = os.path.dirname(__file__)
sys.path.append(root)

from chatdev.chat_chain import ChatChain
from camel.model_backend import ModelBackend

try:
    from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall
    from openai.types.chat.chat_completion_message import FunctionCall

    openai_new_api = True  # new openai api version
except ImportError:
    openai_new_api = False  # old openai api version
    print(
        "Warning: Your OpenAI version is outdated. \n "
        "Please update as specified in requirement.txt. \n "
        "The old API interface is deprecated and will no longer be supported.")


def delete_directory_contents(path):
    os.makedirs(path, exist_ok=True)
    # Ensure the path exists and is a directory
    if not os.path.isdir(path):
        print(f"The provided path {path} does not exist or is not a directory.")
        return

    # List all files and subdirectories in the given directory
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)

        try:
            # If it's a file or a symbolic link, delete it
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
                # print(f"Deleted file: {file_path}")
            # If it's a directory, delete it and all its contents
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
                # print(f"Deleted directory: {file_path}")
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='argparse')
    parser.add_argument('--config', type=str, default="base_config",
                        help="Name of config, which is used to load configuration under CompanyConfig/")
    parser.add_argument('--org', type=str, default="DefaultOrganization",
                        help="Name of organization, your software will be generated in WareHouse/name_org_timestamp")
    # parser.add_argument('--task', type=str, default="Develop a basic Gomoku game.",
    #                     help="Prompt of software")
    # parser.add_argument('--name', type=str, default="Gomoku",
    #                     help="Name of software, your software will be generated in WareHouse/name_org_timestamp")
    parser.add_argument('--model', type=str, default="GPT_3_5_TURBO",
                        help="GPT Model, choose from {'GPT_3_5_TURBO', 'GPT_4', 'GPT_4_TURBO', 'GPT_4O', 'GPT_4O_MINI'}")
    parser.add_argument('--coding_model', type=str, default="llama70b",
                        help="")
    parser.add_argument('--coding_sampling', type=int, default=1,
                        help="")
    parser.add_argument('--review_model', type=str, default="llama70b",
                        help="")
    parser.add_argument('--review_sampling', type=int, default=1,
                        help="")
    parser.add_argument('--test_model', type=str, default="llama70b",
                        help="")
    parser.add_argument('--test_sampling', type=int, default=1,
                        help="")
    parser.add_argument('--examples', type=int, default=5,)

    parser.add_argument('--path', type=str, default="",
                        help="Your file directory, ChatDev will build upon your software in the Incremental mode")
    parser.add_argument('--test', action='store_true', help="test or train.")
    args = parser.parse_args()

    if args.test:
        args.examples = 100

    wandb.init(project="ChatDev", config=args)
    args = wandb.config
    warehouse_dir = "WareHouse" + str(int(time.time()))
    delete_directory_contents(warehouse_dir)

    # Start ChatDev

    # ----------------------------------------
    #          Init ChatChain
    # ----------------------------------------
    config_path, config_phase_path, config_role_path = get_config(args.config)

    ### 修改这里使其变成 (1) 从base config中加载，（2）然后根据args的参数生成新的configs，（3）保存到本地，（4）把路径传给下面，（5）保存wandb的结果
    with open(config_path, 'r', encoding="utf8") as file:
        config = json.load(file)
    with open(config_phase_path, 'r', encoding="utf8") as file:
        config_phase = json.load(file)
    with open(config_role_path, 'r', encoding="utf8") as file:
        config_role = json.load(file)

    coding_sampling = args.coding_sampling
    review_sampling = args.review_sampling
    test_sampling = args.test_sampling

    config_phase['Coding']['repeating'] = coding_sampling
    config_phase['CodeComplete']['repeating'] = coding_sampling
    config_phase['CodeReviewComment']['repeating'] = review_sampling
    config_phase['CodeReviewModification']['repeating'] = review_sampling
    config_phase['TestErrorSummary']['repeating'] = test_sampling
    config_phase['TestModification']['repeating'] = test_sampling
    # config['chain'][3]['cycleNum'] = coding_sampling
    # config['chain'][4]['cycleNum'] = review_sampling
    # config['chain'][5]['cycleNum'] = test_sampling
    coding_model = args.coding_model
    review_model = args.review_model
    test_model = args.test_model
    config_phase['Coding']['model'] = coding_model
    config_phase['CodeComplete']['model'] = coding_model
    config_phase['CodeReviewComment']['model'] = review_model
    config_phase['CodeReviewModification']['model'] = review_model
    config_phase['TestErrorSummary']['model'] = test_model
    config_phase['TestModification']['model'] = test_model

    def budget_compute(m1, n1, m2, n2, m3, n3):
        def trans_m_to_n(m):
            if '70b' in m:
                d = 1
            elif '8b' in m:
                d = 32
            elif '3b' in m:
                d = 90
            return d
        d1 = trans_m_to_n(m1)
        d2 = trans_m_to_n(m2)
        d3 = trans_m_to_n(m3)
        budget = n1/d1 + n2/d2 + n3/d3
        return budget
    # if '8b' not in coding_model and '8b' not in review_model and '8b' not in test_model:
    #     wandb.finish()
    #     print('no 8b!!!')
    #     sys.exit(0)
    budget = budget_compute(coding_model, coding_sampling, review_model, review_sampling, test_model, test_sampling)
    if budget > 11:
         wandb.finish()
         print('budget is not enough!!!')
         sys.exit(0)






    config_dir = os.path.join(root, "CompanyConfig")
    dir_name = f"test_repeating_Coding{coding_model}_{coding_sampling}_Review{review_model}_{review_sampling}_Test_{test_model}_{test_sampling}"
    dir_full_path = os.path.join(config_dir, dir_name)
    if not os.path.exists(dir_full_path):
        os.makedirs(dir_full_path)
    with open(os.path.join(dir_full_path, "ChatChainConfig.json"), 'w', encoding="utf8") as file:
        json.dump(config, file, indent=2)
    with open(os.path.join(dir_full_path, "PhaseConfig.json"), 'w', encoding="utf8") as file:
        json.dump(config_phase, file, indent=2)
    with open(os.path.join(dir_full_path, "RoleConfig.json"), 'w', encoding="utf8") as file:
        json.dump(config_role, file, indent=2)
    # args.config = dir_name
    config_path, config_phase_path, config_role_path = get_config(dir_name)
    ####


    args2type = {'GPT_3_5_TURBO': ModelType.GPT_3_5_TURBO,
                 'GPT_4': ModelType.GPT_4,
                #  'GPT_4_32K': ModelType.GPT_4_32k,
                 'GPT_4_TURBO': ModelType.GPT_4_TURBO,
                #  'GPT_4_TURBO_V': ModelType.GPT_4_TURBO_V
                'GPT_4O': ModelType.GPT_4O,
                'GPT_4O_MINI': ModelType.GPT_4O_MINI,  # is also a string
                 'llama1b': ModelType.LLAMA_1B,
                 'llama3b': ModelType.LLAMA_3B,
                 'llama8b': ModelType.LLAMA_8B,
                 'llama70b': ModelType.LLAMA_70B,
                 'llama405b': ModelType.LLAMA_405B,
                 'qwen3b': ModelType.QWEN_3B,
                 'qwen7b': ModelType.QWEN_7B,
                 'qwen14b': ModelType.QWEN_14B,
                 'qwen32b': ModelType.QWEN_32B,
                 'qwen72b': ModelType.QWEN_72B,
                 }
    # args2type_copy = args2type.copy()
    # for key, value in args2type.items():
    #     for i in range(1, 101):
    #         args2type_copy[f"{key}-{i}"] = value
    # args2type = args2type_copy

    if openai_new_api:
        args2type['GPT_3_5_TURBO'] = ModelType.GPT_3_5_TURBO_NEW

    example_num = args.examples
    if not args.test: # non-test
        with open('SRDD/tasks.json', 'r', encoding="utf8") as file:
            data = json.load(file)
        tasks = data['tasks'][:example_num]
        names = data['names'][:example_num]
    else: # test
        with open('SRDD/tasks.json', 'r', encoding="utf8") as file:
            data = json.load(file)
        tasks = data['tasks'][:example_num]
        names = data['names'][:example_num]


    stats_total = {"instance_count":{}, "method_calls_count":{}}
    count = 0
    for name, task in zip(names, tasks):
        print(count+1, name, task)
        try:
            count += 1
            chat_chain = ChatChain(config_path=config_path,
                                   config_phase_path=config_phase_path,
                                   config_role_path=config_role_path,
                                   task_prompt=task,
                                   project_name=name,
                                   org_name=args.org,
                                   model_type=args2type[args.model],
                                   code_path=args.path,
                                   warehouse=warehouse_dir)

            # ----------------------------------------
            #          Init Log
            # ----------------------------------------
            # print('Log,', chat_chain.log_filepath)
            # Remove all handlers associated with the root logger
            for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)
            logging.basicConfig(filename=chat_chain.log_filepath, level=logging.INFO,
                                format='[%(asctime)s %(levelname)s] %(message)s',
                                datefmt='%Y-%d-%m %H:%M:%S', encoding="utf-8")

            with open(chat_chain.log_filepath, 'w') as file:
                file.write("   \n")
            # ----------------------------------------
            #          Pre Processing
            # ----------------------------------------

            chat_chain.pre_processing()

            # ----------------------------------------
            #          Personnel Recruitment
            # ----------------------------------------

            chat_chain.make_recruitment()

            # ----------------------------------------
            #          Chat Chain
            # ----------------------------------------

            chat_chain.execute_chain()

            # ----------------------------------------
            #          Post Processing
            # ----------------------------------------

            chat_chain.post_processing()

            n1, n2, n3 = chat_chain.config['chain'][3]['cycleNum'], chat_chain.config['chain'][4]['cycleNum'], chat_chain.config['chain'][5]['cycleNum'],
            cc_model, crc_model, tes_model = chat_chain.config_phase['CodeComplete'], chat_chain.config_phase['CodeReviewComment'], chat_chain.config_phase['TestErrorSummary']

            # stats = ModelBackend.get_stats()
            # instance_count = stats['instance_count']
            # method_calls_count = stats['method_calls_count']
            # for key, value in instance_count.items():
            #     if key in stats_total['instance_count']:
            #         stats_total['instance_count'][key] += value
            #     else:
            #         stats_total['instance_count'][key] = value
            # for key, value in method_calls_count.items():
            #     if key in stats_total['method_calls_count']:
            #         stats_total['method_calls_count'][key] += value
            #     else:
            #         stats_total['method_calls_count'][key] = value

            # stats['cycleNum'] = (n1, n2, n3)
            # stats['model_choices'] = cc_model, crc_model, tes_model
            # print('Log, stats:', stats)
        except Exception as e:

            print(count+1, 'something went wrong')
            continue

    stats = ModelBackend.get_stats()
    print(stats)
    wandb.log(stats)

    # path = os.path.join(root + "/logs", "_".join([args.config, args.name, args.org, chat_chain.start_time]))
    # os.makedirs(path, exist_ok=True)
    # file = os.path.join(path, "run_stats.json")
    # with open(file, "w") as f:
    #     json.dump(stats, f, indent=2)

    results_dict = eval(warehouse_dir, dir_name)
    print("eval results:", results_dict)
    wandb.log(results_dict)
    wandb.finish()

'''
test script: 
nohup python run_repeating.py --config base_config --coding_model llama70b --coding_sampling 2 --review_model llama3b --review_sampling 60 --test_model llama3b --test_sampling 90 --org SRDD_Action_Game --test > log_chatdev_test1.log 2>&1 &
nohup python run_repeating.py --config base_config --coding_model llama3b --coding_sampling 70 --review_model llama3b --review_sampling 30 --test_model llama70b --test_sampling 1 --org SRDD_Action_Game --test > log_chatdev_test2.log 2>&1 &
nohup python run_repeating.py --config base_config --coding_model llama70b --coding_sampling 3 --review_model llama3b --review_sampling 2 --test_model llama70b --test_sampling 1 --org SRDD_Action_Game --test > log_chatdev_test3.log 2>&1 &
nohup python run_repeating.py --config base_config --coding_model llama70b --coding_sampling 2 --review_model llama3b --review_sampling 70 --test_model llama3b --test_sampling 70 --org SRDD_Action_Game --test > log_chatdev_test4.log 2>&1 &
nohup python run_repeating.py --config base_config --coding_model llama3b --coding_sampling 30 --review_model llama3b --review_sampling 60 --test_model llama3b --test_sampling 1 --org SRDD_Action_Game --test > log_chatdev_test5.log 2>&1 &
nohup python run_repeating.py --config base_config --coding_model llama70b --coding_sampling 3 --review_model llama3b --review_sampling 50 --test_model llama3b --test_sampling 1 --org SRDD_Action_Game --test > log_chatdev_test6.log 2>&1 &

best config: 
# best config
# ours: code llama70b 2 review llama3b 60 test llama3b 90 0.748
# hpo: code llama3b 70 review llama3b 30 test llama70b 1 0.740
# copilot: code llama70b 3 review llama3b 2 test llama70b 1 0.752
# zs: code llama70b 2 review llama3b 70 test llama3b 70 0.738
# bayes: code llama3b 30 review llama3b 60 test llama3b 1 0.752
# random: code llama70b 3 review llama3b 50 test llama3b 1 0.741 

'''
