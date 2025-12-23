import os
import torch
import codecs
import json
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig, get_peft_model

# =========================
# 1. Config
# =========================
RUN_ON_MAC = True

if RUN_ON_MAC:
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.set_default_dtype(torch.float32)

LOAD_MODEL_FROM_CACHE = True
LOAD_DATASET_DEMO = False

MODEL_ID = "qwen3/Qwen3-0.6B"
RESTORE_MODEL_DIR_CACHE = "../train/model/qwen3/Qwen3-0.6B/Qwen/Qwen3-0___6B"
MODEL_PATH = RESTORE_MODEL_DIR_CACHE if LOAD_MODEL_FROM_CACHE else MODEL_ID


OUTPUT_DIR = "../train/model/grpo_qwen3_rl_trl"

output_dataset_demo_path = "../train/dataset/rl/train_demo.parquet"
input_dataset_path = "../train/dataset/rl/example_deepnlp_rl_feedback_202510.json"
output_dataset_path = "../train/dataset/rl/train.parquet"
DATASET_PATH = output_dataset_demo_path if LOAD_DATASET_DEMO else output_dataset_path

# 1. Configuration & Mac Setup
# 2. Dataset Preparation (Tool Calling Format)
def prepare_tool_dataset_demo():
    """
        # verl/trl conversational format + direct scalar rewards
    """
    # verl expects: data_source, prompt, ability, and reward_model (ground_truth)
    raw_data = [
        {
            "data_source": "tool_call_weather",
            "prompt": [{"role": "user", "content": "What is the weather in SF?"}],
            "ability": "tool_call",
            "reward_model": {
                "style": "rule",
                "ground_truth": "get_weather(city='SF')"
            }
        },
        {
            "data_source": "tool_call_math",
            "prompt": [{"role": "user", "content": "Calculate 5 + 5 using the calculator tool."}],
            "ability": "tool_call",
            "reward_model": {
                "style": "rule",
                "ground_truth": "calculator(add=[5,5])"
            }
        }
    ]

    raw_data_list = raw_data * 100
    print (f"DEBUG: raw_data_list size {len(raw_data_list)}")
    dataset = Dataset.from_list(raw_data_list)
    # verl usually reads from parquet
    dataset.to_parquet(output_dataset_demo_path)
    print(f"Dataset saved to {output_dataset_demo_path}")

def extract_prompt(example):
    return {
        "prompt": example["prompt"][0]["content"],
        "ground_truth": example["reward_model"]["ground_truth"],
    }

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

def prepare_tool_dataset_scalar_rewards():
    """
        {
            "prompt":
                [{'role': 'user',
                  'content': "Draw a pie chart of these data, display percentage not raw number. {'finance': 79, 'AI AGENT': 304, 'knowledge-and-memory': 98, 'Audio Generator': 53, 'Heathcare': 22, 'other': 93, 'art-and-culture': 13, 'BUSINESS': 12, 'AI Agent Marketplace': 38, 'calendar-management': 140, 'Search Agent': 88, 'Travel': 85, 'RESEARCH': 68, 'AI Search Agent': 35, 'AI AGENT PLATFORM': 41, 'AI Tutor Agent': 31, 'Recommendation Agent': 14, 'BENCHMARK': 21, 'browser-automation': 87, 'Trip Planning': 19, 'Coding Agent': 69, 'Robotaxi': 23, 'WORKFLOW': 73, 'Tool Agent': 62, 'Data Analysis': 21, 'CHATBOT': 93, 'AI Docs': 26, 'Email Writing': 65, 'AI Agent Tool': 18, 'Assistant': 40, 'CODING AGENT': 217, 'Ads': 33, 'Music Generator': 115, 'Science': 16, 'Quadruped Robot': 34, 'ASSISTANT': 63, 'Marketing': 18, 'AI Law Agent': 17, 'AI AGENTS FRAMEWORKS': 108, 'Humanoid Robot': 34, 'file-systems': 60, 'Website Builder': 33, 'entertainment-and-media': 51, 'Developer': 8, 'AI Agent Framework': 7, 'Healthcare': 39, 'Orchestrating Agents': 36, 'Image Generator': 50, 'Law': 36, 'AI4Science Agent': 38, 'TRANSLATION': 13, 'FINANCE': 19, 'SCIENCE': 19, 'Tool Libraries': 29, 'Recruiting': 30, 'VIDEO GENERATOR': 3, 'developer-tools': 34, 'communication': 100, 'Search Engine Agent': 30, 'Entertainment': 16, 'Legal': 5, 'AI Agent Employees': 19, 'AI Agent Teacher': 16, 'Business': 18, 'Chatbot': 41, 'Deep Research Agent': 11, 'Education': 22, 'Benchmark': 13, 'Software Testing': 30, 'Sales': 23, 'AI Agent Healthcare': 33, 'Embodied AI': 41, 'research-and-data': 41, 'search': 18, 'Productivity': 15, 'AI Agent Finance': 11, 'DESIGN': 3, 'Desktop Use': 15, 'AI Shopping': 64, 'PRODUCTIVITY': 9, 'AI SECURITY': 12, 'Video Generator': 58, 'AI Agent': 50, 'Gaming Agent': 25, 'AI Agent Directory': 11, 'Search Recommendation': 9, 'Digital Workers': 23, 'AI Security': 19, 'Research': 19, 'Blog Writing': 11, 'AI Agent Platform': 20, 'AI Agent Legal': 22, 'AI AGENT MARKETPLACE': 2, 'location-services': 27, 'AI Agent Employee': 12, 'AI Agent Memory': 14, 'AI HR Agent': 1, 'AI Agent Orchestration': 9, 'AI Agents Frameworks': 40, 'AI Agent Search': 10, 'SOCIAL': 1, 'AI Employee': 31, 'SALES': 13, 'Customer Service': 42, 'BLOG WRITING': 3, 'EMBODIED AI': 4, 'AI Gadget': 7, 'Mobile Use': 3, 'HR Agent': 12, 'FILE': 2, 'File System': 1, 'Database': 4, 'Finance': 13, 'PAYMENT': 2, 'AI DOCS': 3, 'AI SHOPPING': 1, 'Workflow': 6, 'WEB': 2, 'AI AGENT MEMORY': 12, 'AI AGENT BUILDER': 9, 'EDUCATION': 3, 'AUTONOMOUS AGENT': 2, 'Workplace': 15, 'Operations': 8, 'ADVERTISING': 1, 'Gaming Agents': 8, 'MAP': 2, 'Translation': 14, 'Browser': 1, 'BLOCKCHAIN': 1, 'Autonomous Agent': 5, 'Workflow ': 1, 'DATA': 1, 'HEALTHCARE': 4, 'IMAGE GENERATOR': 3, 'AI Agent Frameworks': 4, 'MARKETING': 2, 'SEARCH': 1, 'AI Entertainment Agent': 2, 'AI Agent Law': 4, 'AI AGENT DIRECTORY': 1, 'Official': 1, 'COPILOT AGENT': 1, 'DATABASE': 1, 'ENTERTAINMENT': 1, 'AI Agent Index': 1, 'Map': 1, 'Memory': 2, 'Thinking': 1, 'AI Operations Agent': 2, 'Browser Use': 1}"},
                 {'role': 'user',
                  'content': '<div class="segment-container"><div class="file-container-grid">\n                                        <div class="message_file_display_wrapper" data-file-name="employees.xlsx" data-file-url="/files-wd/download/demo/demo_session_id/employees.xlsx" data-file-type="xlsx">\n                                            <div class="file_display_icon_wrapper">\n                                                <img class="file_display_icon" src="/static/img/excel-icon-transparent.png"></img>\n                                            </div>\n                                            <div>\n                                                <span class="file-name">employees.xlsx</span>\n                                            </div>\n                                        </div>\n                                    </div></div>'},
                 {'role': 'assistant',
                  'content': '<img class="agent_display_icon" src="https://agent.deepnlp.org/static/img/icon_chatbot_agent.png"> Multi Agents Start Running<br>Agent (plan_agent) Start Planning Tasks<br>Agent (base_agent) Execution Start Tasks <br><p>Iteration 1 Running Tool <span class="span_planning_highlight">generate_pie_chart</span> of MCP Server <span class="span_planning_highlight">mcp-server-chart</span></p><br><p>Iteration 2 Running Tool <span class="span_planning_highlight">generate_pie_chart</span> of MCP Server <span class="span_planning_highlight">mcp-server-chart</span></p><br><p>Iteration 3 Running Tool <span class="span_planning_highlight">generate_pie_chart</span> of MCP Server <span class="span_planning_highlight">mcp-server-chart</span></p><br><br>Agent (base_agent) Execution End <br>Agent (generation_agent) Start Generating Answers<br>\n<url>https://mdn.alipayobjects.com/one_clip/afts/img/JEeRRbKd44wAAAAAYzAAAAgAoEACAQFr/original</url>'}],
            "rewards": 1.0
        }
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

            data_list.append({
                "data_source": "message_history",
                "prompt": messages,
                "reward": reward,
                "reward_description": reward_description
            })
        except Exception as e:
            print (f"Failed to process {e}")
    ## 
    dataset = Dataset.from_list(data_list)
    
    # Save to Parquet
    dataset.to_parquet(output_dataset_path)
    print(f"Scalar Dataset saved to {output_dataset_path}")

# =========================
# 4. Reward Function (TRL-style)
# =========================

def tool_call_reward_fn(prompts, completions, ground_truth, **kwargs):
    """
    TRL GRPO reward signature:
      - prompts: List[str]
      - completions: List[str]
      - ground_truth: List[str]


      "tool_calls" are the ke in the generated completion

    """
    rewards = []

    for completion, gt in zip(completions, ground_truth):
        if gt in completion:
            rewards.append(1.0)
        elif "tool_calls" in completion.lower():
            rewards.append(0.2)
        else:
            rewards.append(0.0)

    return rewards


def scalar_lookup_reward_fn(completions, reward, **kwargs):
    """
    completions: The text generated by the model during RL
    reward: The 'reward' column from your dataset
    """
    # In this logic, we are rewarding the model for matching 
    # the 'quality level' defined in the dataset for that specific prompt.
    # Typically, you'd combine this with a formatting check.
    
    rewards = []
    for content, score in zip(completions, reward):
        # Example: 1.0 reward if the model output is valid AND the prompt was 'high quality'
        if len(content) > 0:
            rewards.append(float(score))
        else:
            rewards.append(0.0)
            
    return rewards

# =========================
# 3. Dataset
# =========================

def rl_trl_grpo_trainer_on_scalar_reward():
    """ RL TRL GRPO Trainer
    """
    ## keys: 
    ## "prompt", "rewards"
    dataset = load_dataset("parquet", data_files=DATASET_PATH)["train"]
    ## prompt, ground_truth
    if LOAD_DATASET_DEMO:
        dataset = dataset.map(extract_prompt, remove_columns=dataset.column_names)

    # =========================
    # 1. Model & Tokenizer
    # =========================
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    if RUN_ON_MAC:
        model = AutoModelForCausalLM.from_pretrained(
                    MODEL_PATH,
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                    device_map=None, 
        )
        model.config.torch_dtype = torch.float32
        model.to("mps")
    else:
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Wrap your existing float32 model
    model = get_peft_model(model, lora_config)

    # =========================
    # 6. GRPO Configuration
    # =========================
    config = GRPOConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        # per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        # GRPO-specific
        generation_batch_size=8,
        num_generations=4,            # GROUP SIZE
        max_completion_length=64,
        temperature=0.9,
        # Optimization
        learning_rate=2e-6,
        # kl_coef=0.05,
        beta = 0.05,
        # Logging
        logging_steps=1,
        save_steps=50,
        report_to="none",

        # MPS safety
        bf16=False,
        fp16=False,
    )

    # =========================
    # 7. Trainer
    # =========================

    ##  Reference Model might load BF16 again
    ## Add LORA , Demo, Function Call + Rewards
    # trainer = GRPOTrainer(
    #     model=model,
    #     processing_class=tokenizer,
    #     args=config,
    #     train_dataset=dataset,
    #     reward_funcs=[tool_call_reward_fn],
    # )
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=config,
        train_dataset=dataset,
        reward_funcs=[tool_call_reward_fn] if LOAD_DATASET_DEMO else [scalar_lookup_reward_fn]
    )

    # =========================
    # 8. Train
    # =========================
    trainer.train()

    # =========================
    # 9. Save
    # =========================

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"\n✅ GRPO training complete. Model saved to {OUTPUT_DIR}")


def rl_trl_grpo_trainer_on_tool_call():
    """
        Input: prompt + tool_call
        [
            {
                "data_source": "tool_call_weather",
                "prompt": [{"role": "user", "content": "What is the weather in SF?"}],
                "ability": "tool_call",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": "get_weather(city='SF')"
                }
            }
        ]

        Evaluate weather the prompt and generated tool_call, such as 'get_weather(city='SF')' is correct or not.
    """

    return

def main():
    ## prepare demo dataset， tool_call ground_truth
    prepare_tool_dataset_demo()
    ## Download Datasets From Web
    prepare_tool_dataset_scalar_rewards()
    ## Train RL TRL GRPO Trainer
    rl_trl_grpo_trainer_on_scalar_reward()

if __name__ == '__main__':
    main()
