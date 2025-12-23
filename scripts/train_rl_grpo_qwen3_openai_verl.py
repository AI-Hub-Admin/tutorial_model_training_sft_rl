import codecs
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk
from datasets import Dataset
from typing import List, Any, Dict

### Restore and download model from modelscope if huggingface is not connected

LOAD_FROM_CACHE = True
restore_model_dir = "../train/model/qwen3/Qwen3-0.6B"
restore_model_dir_cache = "../train/model/qwen3/Qwen3-0.6B/Qwen/Qwen3-0___6B"
output_sft_model_dir = "../train/model/sft/grpo_qwen3_rl_verl"
DATASET_PATH = "../train/dataset/rl/verl_train.parquet"


output_dataset_demo_path = "../train/dataset/rl/verl_train_demo.parquet"
input_dataset_path = "../train/dataset/rl/example_deepnlp_rl_feedback_202510.json"
output_dataset_path = "../train/dataset/rl/verl_train.parquet"

LOAD_DATASET_DEMO = False
DATASET_PATH = output_dataset_demo_path if LOAD_DATASET_DEMO else output_dataset_path


def render_messages(messages):
    """Convert OpenAI-style messages to plain text."""
    lines = []
    for m in messages:
        role = m["role"]
        if role == "user":
            lines.append(f"User: {m['content']}")
        elif role == "assistant":
            if "content" in m and m["content"]:
                lines.append(f"Assistant: {m['content']}")
            elif "tool_calls" in m:
                for tc in m["tool_calls"]:
                    lines.append(
                        f"Assistant (tool call): {tc['function']['name']}("
                        f"{tc['function']['arguments']})"
                    )
        elif role == "tool":
            lines.append(f"Tool result: {m['content']}")
    return "\n".join(lines)

def to_grpo(example: Dict):
    messages = example["messages"]

    prompt_msgs = [m for m in messages if m["role"] == "user"]
    response_msgs = [m for m in messages if m["role"] != "user"]

    return {
        "prompt": render_messages(prompt_msgs),
        "response": render_messages(response_msgs),
        "reward": float(example["reward"]),
    }

def prepare_rl_dataset_verl_demo():
    raw_data = [
        {
            "messages": [
                {"role": "user", "content": "What's the weather in SF?"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": "{\"city\": \"SF\"}"
                            }
                        }
                    ]
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": "{\"temperature\": 18}"
                },
                {
                    "role": "assistant",
                    "content": "The temperature in San Francisco is 18°C."
                }
            ],
            "reward": 1.0
        }
    ]

    grpo_dataset = Dataset.from_list([to_grpo(x) for x in raw_data])
    print(f"Verl Dataset Demo saved to {output_dataset_demo_path}")    
    grpo_dataset.save_to_disk(output_dataset_demo_path)

def read_file(file_path):
    lines = []
    lines_clean = []
    try:
        with codecs.open(file_path, "r", "utf-8") as file:
            lines = file.readlines()
        for line in lines:
            line_clean = line.strip()
            line_clean = line_clean.replace("\n", "")
            lines_clean.append(line_clean)
    except Exception as e:
        print ("DEBUG: read_file failed file_path %s" % file_path)
        print (e)
    return lines_clean


def prepare_rl_dataset_verl():
    """
        GRPO Format VERL

        {'prompt': 'User: Draw a pie chart of these data, display percentage not raw number. {\'finance\': 79, \'AI AGENT\': 304, \'knowledge-and-memory\': 98, \'Audio Generator\': 53, \'Heathcare\': 22, \'other\': 93, \'art-and-culture\': 13, \'BUSINESS\': 12, \'AI Agent Marketplace\': 38, \'calendar-management\': 140, \'Search Agent\': 88, \'Travel\': 85, \'RESEARCH\': 68, \'AI Search Agent\': 35, \'AI AGENT PLATFORM\': 41, \'AI Tutor Agent\': 31, \'Recommendation Agent\': 14, \'BENCHMARK\': 21, \'browser-automation\': 87, \'Trip Planning\': 19, \'Coding Agent\': 69, \'Robotaxi\': 23, \'WORKFLOW\': 73, \'Tool Agent\': 62, \'Data Analysis\': 21, \'CHATBOT\': 93, \'AI Docs\': 26, \'Email Writing\': 65, \'AI Agent Tool\': 18, \'Assistant\': 40, \'CODING AGENT\': 217, \'Ads\': 33, \'Music Generator\': 115, \'Science\': 16, \'Quadruped Robot\': 34, \'ASSISTANT\': 63, \'Marketing\': 18, \'AI Law Agent\': 17, \'AI AGENTS FRAMEWORKS\': 108, \'Humanoid Robot\': 34, \'file-systems\': 60, \'Website Builder\': 33, \'entertainment-and-media\': 51, \'Developer\': 8, \'AI Agent Framework\': 7, \'Healthcare\': 39, \'Orchestrating Agents\': 36, \'Image Generator\': 50, \'Law\': 36, \'AI4Science Agent\': 38, \'TRANSLATION\': 13, \'FINANCE\': 19, \'SCIENCE\': 19, \'Tool Libraries\': 29, \'Recruiting\': 30, \'VIDEO GENERATOR\': 3, \'developer-tools\': 34, \'communication\': 100, \'Search Engine Agent\': 30, \'Entertainment\': 16, \'Legal\': 5, \'AI Agent Employees\': 19, \'AI Agent Teacher\': 16, \'Business\': 18, \'Chatbot\': 41, \'Deep Research Agent\': 11, \'Education\': 22, \'Benchmark\': 13, \'Software Testing\': 30, \'Sales\': 23, \'AI Agent Healthcare\': 33, \'Embodied AI\': 41, \'research-and-data\': 41, \'search\': 18, \'Productivity\': 15, \'AI Agent Finance\': 11, \'DESIGN\': 3, \'Desktop Use\': 15, \'AI Shopping\': 64, \'PRODUCTIVITY\': 9, \'AI SECURITY\': 12, \'Video Generator\': 58, \'AI Agent\': 50, \'Gaming Agent\': 25, \'AI Agent Directory\': 11, \'Search Recommendation\': 9, \'Digital Workers\': 23, \'AI Security\': 19, \'Research\': 19, \'Blog Writing\': 11, \'AI Agent Platform\': 20, \'AI Agent Legal\': 22, \'AI AGENT MARKETPLACE\': 2, \'location-services\': 27, \'AI Agent Employee\': 12, \'AI Agent Memory\': 14, \'AI HR Agent\': 1, \'AI Agent Orchestration\': 9, \'AI Agents Frameworks\': 40, \'AI Agent Search\': 10, \'SOCIAL\': 1, \'AI Employee\': 31, \'SALES\': 13, \'Customer Service\': 42, \'BLOG WRITING\': 3, \'EMBODIED AI\': 4, \'AI Gadget\': 7, \'Mobile Use\': 3, \'HR Agent\': 12, \'FILE\': 2, \'File System\': 1, \'Database\': 4, \'Finance\': 13, \'PAYMENT\': 2, \'AI DOCS\': 3, \'AI SHOPPING\': 1, \'Workflow\': 6, \'WEB\': 2, \'AI AGENT MEMORY\': 12, \'AI AGENT BUILDER\': 9, \'EDUCATION\': 3, \'AUTONOMOUS AGENT\': 2, \'Workplace\': 15, \'Operations\': 8, \'ADVERTISING\': 1, \'Gaming Agents\': 8, \'MAP\': 2, \'Translation\': 14, \'Browser\': 1, \'BLOCKCHAIN\': 1, \'Autonomous Agent\': 5, \'Workflow \': 1, \'DATA\': 1, \'HEALTHCARE\': 4, \'IMAGE GENERATOR\': 3, \'AI Agent Frameworks\': 4, \'MARKETING\': 2, \'SEARCH\': 1, \'AI Entertainment Agent\': 2, \'AI Agent Law\': 4, \'AI AGENT DIRECTORY\': 1, \'Official\': 1, \'COPILOT AGENT\': 1, \'DATABASE\': 1, \'ENTERTAINMENT\': 1, \'AI Agent Index\': 1, \'Map\': 1, \'Memory\': 2, \'Thinking\': 1, \'AI Operations Agent\': 2, \'Browser Use\': 1}\nUser: <div class="segment-container"><div class="file-container-grid">\n                                        <div class="message_file_display_wrapper" data-file-name="employees.xlsx" data-file-url="/files-wd/download/demo/demo_session_id/employees.xlsx" data-file-type="xlsx">\n                                            <div class="file_display_icon_wrapper">\n                                                <img class="file_display_icon" src="/static/img/excel-icon-transparent.png"></img>\n                                            </div>\n                                            <div>\n                                                <span class="file-name">employees.xlsx</span>\n                                            </div>\n                                        </div>\n                                    </div></div>',
         'response': 'Assistant: <img class="agent_display_icon" src="https://agent.deepnlp.org/static/img/icon_chatbot_agent.png"> Multi Agents Start Running<br>Agent (plan_agent) Start Planning Tasks<br>Agent (base_agent) Execution Start Tasks <br><p>Iteration 1 Running Tool <span class="span_planning_highlight">generate_pie_chart</span> of MCP Server <span class="span_planning_highlight">mcp-server-chart</span></p><br><p>Iteration 2 Running Tool <span class="span_planning_highlight">generate_pie_chart</span> of MCP Server <span class="span_planning_highlight">mcp-server-chart</span></p><br><p>Iteration 3 Running Tool <span class="span_planning_highlight">generate_pie_chart</span> of MCP Server <span class="span_planning_highlight">mcp-server-chart</span></p><br><br>Agent (base_agent) Execution End <br>Agent (generation_agent) Start Generating Answers<br>\n<url>https://mdn.alipayobjects.com/one_clip/afts/img/JEeRRbKd44wAAAAAYzAAAAgAoEACAQFr/original</url>',
         'reward': 1.0}

    """
    # verl expects: data_source, prompt, ability, and reward_model (ground_truth)
    lines = read_file(input_dataset_path)
    print (f"DEBUG: Reading Dataset Lines {len(lines)}")

    data_list = []
    ## keys: ['model', 'session_id', 'trace_id', 'messages', 'reward', 'reward_description']
    for line in lines:
        try:
            item_json = json.loads(line)
            ## merge messages and tool_calls into a prompt list
            messages = item_json.get("messages", [])
            reward = item_json.get("reward", 0.0)
            reward_description = item_json.get("reward_description", "")
            ## Convert Dict of messages to grpo formats
            grpo_example =to_grpo(item_json) 
            data_list.append(grpo_example)

        except Exception as e:
            print (f"Failed to process {e}")
    ## 
    grpo_dataset = Dataset.from_list(data_list)
    # Save to Parquet
    print(f"Verl Dataset saved to {output_dataset_path}")
    grpo_dataset.save_to_disk(output_dataset_demo_path)

def rl_verl_grpo_trainer():
    """
        {
            "prompt": "User-visible prompt text",
            "response": "Assistant-visible response text",
            "reward": float
        }
    """
    from verl.trainer import GRPOTrainer
    ### load model
    if LOAD_FROM_CACHE:
        tokenizer = AutoTokenizer.from_pretrained(restore_model_dir_cache)
        tokenizer.pad_token = tokenizer.eos_token
        if run_on_mac:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
                trust_remote_code=True
            )
            model.to("mps")
        else:
            model = AutoModelForCausalLM.from_pretrained(model_path)
    else:
        # 1. Load Tokenizer and Model
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            # torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
    ## 
    dataset = load_from_disk(DATASET_PATH)

    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        max_prompt_length=4096,
        max_response_length=4096,
        kl_coef=0.05,
        learning_rate=5e-6,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
    )

    trainer.train()

def main():
    ## Prepare Datasets
    prepare_rl_dataset_verl_demo()

    prepare_rl_dataset_verl()
    ## verl starts from the CLI
    # rl_verl_grpo_trainer()

if __name__ == '__main__':
    main()
