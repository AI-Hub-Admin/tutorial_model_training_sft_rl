import json
from datasets import Dataset, Features, Sequence, Value

import codecs
import json
import re
import uuid

demo_dataset_path = "../train/dataset/function_call/function_calling_dataset_qwen_demo"

### Dataset Path
download_dataset_path = "../train/dataset/function_call/example_deepnlp_agent_function_call_202510.json"
convert_dataset_path = "../train/function_calling_dataset_qwen"

### Restore and download model from modelscope if huggingface is not connected
restore_model_dir = "../train/model/qwen3/Qwen3-0.6B"
restore_model_dir_cache = "../train/model/qwen3/Qwen3-0.6B/Qwen/Qwen3-0___6B"
output_sft_model_dir = "../train/model/sft/qwen3-function-calling"

SYSTEM_PROMPT = """You are a helpful assistant.
    You have access to the following tools:
    <tools>
    {tools_json}
    </tools>
    For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
    <tool_call>
    {{"name": function-name, "arguments": args-json-object}}
    </tool_call>"""


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


def normalize_tool_message(msg):
    # Tool content may be multimodal
    content = msg.get("content", "")

    if isinstance(content, list):
        content = "".join(
            block.get("text", "")
            for block in content
            if block.get("type") == "text"
        )

    return {
        "role": "tool",
        "content": content,
        "tool_calls": [],
        "tool_call_id": msg.get("tool_call_id", ""),
    }

def normalize_message(msg):
    role = msg["role"]

    # TOOL
    if role == "tool":
        return normalize_tool_message(msg)

    # ASSISTANT with tool calls
    if role == "assistant" and "tool_calls" in msg:
        content = msg.get("content", "")
        if isinstance(content, list):
            content = "".join(
                block.get("text", "")
                for block in content
                if block.get("type") == "text"
            )

        return {
            "role": "assistant",
            "content": content or "",
            "tool_calls": msg.get("tool_calls", []),
            "tool_call_id": "",
        }

    # USER / SYSTEM / ASSISTANT (text only)
    content = msg.get("content", "")
    if isinstance(content, list):
        content = "".join(
            block.get("text", "")
            for block in content
            if block.get("type") == "text"
        )

    return {
        "role": role,
        "content": content,
        "tool_calls": [],
        "tool_call_id": "",
    }

def load_download_dataset():
    """
        50 examples: https://static.aiagenta2z.com/scripts/doc/file/06766e91894147319ffd0116b04ff94d/example_deepnlp_agent_function_call_202510.json
        
        Load Datasets
        messages:

            [{'role': 'user', 'content': 'New York Times Square Italian Restaurants'}, {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'call_6107b2e6c70f4c1ea692bf', 'type': 'tool_use', 'function': {'name': 'maps_text_search', 'arguments': '{"keywords": "Italian Restaurants", "city": "New York"}'}}]}, {'role': 'tool', 'tool_call_id': 'call_6107b2e6c70f4c1ea692bf', 'name': 'maps_text_search', 'content': '[{"type": "text", "text": "{\\"suggestion\\":{\\"keywords\\":\\"\\",\\"ciytes\\":{\\"suggestion\\":[]}},\\"pois\\":[]}"}]'}, {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'call_b05c3711efd6445c93208a', 'type': 'tool_use', 'function': {'name': 'maps_text_search', 'arguments': '{"keywords": "Italian Restaurants", "city": "New York City"}'}}]}, {'role': 'tool', 'tool_call_id': 'call_b05c3711efd6445c93208a', 'name': 'maps_text_search', 'content': '[{"type": "text", "text": "{\\"suggestion\\":{\\"keywords\\":\\"\\",\\"ciytes\\":{\\"suggestion\\":[]}},\\"pois\\":[]}"}]'}]
        
        tools: list of json dict

        [{'type': 'function',
          'function': {'name': 'maps_direction_bicycling',
           'description': '骑行路径规划用于规划骑行通勤方案，规划时会考虑天桥、单行线、封路等情况。最大支持 500km 的骑行路线规划',
           'parameters': {'type': 'object',
            'properties': {'origin': {'type': 'string',
              'description': '出发点经纬度，坐标格式为：经度，纬度'},
             'destination': {'type': 'string', 'description': '目的地经纬度，坐标格式为：经度，纬度'}},
            'required': ['origin', 'destination']}}},
         {'type': 'function',
          'function': {'name': 'maps_direction_driving',
           'description': '驾车路径规划 API 可以根据用户起终点经纬度坐标规划以小客车、轿车通勤出行的方案，并且返回通勤方案的数据。',
           'parameters': {'type': 'object',
            'properties': {'origin': {'type': 'string',
              'description': '出发点经纬度，坐标格式为：经度，纬度'},
             'destination': {'type': 'string', 'description': '目的地经纬度，坐标格式为：经度，纬度'}},
            'required': ['origin', 'destination']}}},
            ]
        
        tool_calls:
            {'id': 'call_6107b2e6c70f4c1ea692bf', 
            'function': {'arguments': '{"keywords": "Italian Restaurants", "city": "New York"}', 
            'name': 'maps_text_search'}, 'type': 'function'}


        ## Recommended Formats for ChatGPT
        {
          "messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."},
            {
              "role": "assistant",
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
            }
          ]
        }

        ## Format
        {
          "role": "tool",
          "tool_call_id": "call_b05c3711efd6445c93208a",
          "content": "{\"suggestion\":{\"keywords\":\"\",\"ciytes\":{\"suggestion\":[]}},\"pois\":[]}"
        }
    """
    import json
    files = read_file(download_dataset_path)
    data_example = json.loads(files[0])
    print (f"DEBUG: Loading Data Files Size {len(files)} File Format Example 0 keys {list(data_example.keys())}")

    ### keys: 'messages', 'tools', 'tool_calls'
    function_calls_list = data_example["function_calls"]
    if len(function_calls_list) > 0:
        print (function_calls_list[0].keys())
        messages = function_calls_list[0].get("messages", [])
        tools = function_calls_list[0].get("tools", [])
        tool_calls = function_calls_list[0].get("tool_calls", [])
        print (f"DEBUG: data_example messages {messages}")
        print (f"DEBUG: data_example tools {tools}")
        print (f"DEBUG: data_example tool_calls {tool_calls}")

    ## keys: 
    datalist = []
    for i, line in enumerate(files):
        try:
            data_example = json.loads(line)
            function_calls = data_example["function_calls"]
            for function_call in function_calls:
                messages = function_calls_list[0].get("messages", [])
                tools = function_calls_list[0].get("tools", [])
                tool_calls = function_calls_list[0].get("tool_calls", [])
                ## assembly training dataset
                system_message = {"role": "system", "content": SYSTEM_PROMPT.format(tools_json=tools)}
                messages_merge = [system_message] + messages + [{"role": "assistant", "content": f'<tool_call>{json.dumps(tool_calls)}</tool_call>'}]
                messages_format = [normalize_message(m) for m in messages_merge]

                data_sft = {
                    "messages": messages_format
                }
                datalist.append(data_sft)
        except Exception as e:
            print (f"DEBUG: Failed ro Process Error {e}")

    ## messages formats
    features = Features({
        'messages': [{
            'role': Value('string'),
            'content': Value('string'),
            'tool_calls': [{
                'id': Value('string'),
                'type': Value('string'),
                'function': {
                    'name': Value('string'),
                    'arguments': Value('string')
                }
            }],
            'tool_call_id': Value('string')
        }]
    })

    ## save
    dataset = Dataset.from_list(datalist, features=features)
    dataset.save_to_disk(convert_dataset_path)

def prepare_sample_dataset_qwen():
    import json
    from datasets import Dataset

    # Qwen-specific Tool Schema (Standard JSON)
    weather_tool = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City and state, e.g. San Francisco, CA"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["location"]
            }
        }
    }

    # The system prompt Qwen uses to understand tools
    data = []

    # Example: Standard Weather Call
    tools_list = [weather_tool]
    tools_json = json.dumps(tools_list, indent=2)

    tool_calls = '{"name": "get_weather", "arguments": {"location": "London"}}'

    data.append({
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT.format(tools_json=tools_json)},
            {"role": "user", "content": "What is the weather in London?"},
            {"role": "assistant", "content": f"<tool_call>\n{tool_calls}\n</tool_call>"},
            {"role": "tool", "content": '{"temp": 15, "unit": "celsius"}'},
            {"role": "assistant", "content": "The current weather in London is 15°C."}
        ]
    })


    # Generate 9 more similar examples...
    for i in range(9):
        data.append(data[0])


    dataset = Dataset.from_list(data)
    dataset.save_to_disk(demo_dataset_path)


def sft_trainer_setup():
    """
    ## download data from modelscope
    """
    from modelscope import snapshot_download
    from transformers import AutoModel, AutoTokenizer
    model_path = snapshot_download(
        model_id="Qwen/Qwen3-0.6B",
        cache_dir=restore_model_dir
    )
    ## test_restore
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)

def sft_trainer(dataset_path, model_path):
    """
        pip install -U transformers huggingface_hub
        pip install --upgrade torch torchvision torchaudio

        dataset_path="../train/function_calling_dataset_qwen"
    """
    #### Qwen3 SFT Model 
    # import torch
    from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, TrainingArguments
    from trl import SFTTrainer, SFTConfig
    from datasets import load_from_disk
    from peft import LoraConfig
    import torch

    model_id = "Qwen/Qwen3-0.6B"
    
    load_from_cache = True 
    run_on_mac = True
    if load_from_cache:
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

    # 2. LoRA Configuration (Saves memory and prevents forgetting)
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )


    if run_on_mac:
        import os
        os.environ["ACCELERATE_DISABLE_MIXED_PRECISION"] = "true"
        os.environ["ACCELERATE_MIXED_PRECISION"] = "no"
        os.environ["ACCELERATE_CONFIG_FILE"] = "/dev/null"

    # 3. Training Arguments, Testing on Mac, Change bf16, fp16 when running on GPU
    sft_config = SFTConfig(
        output_dir=output_sft_model_dir,
        dataset_text_field="messages",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        num_train_epochs=3,
        bf16=False,
        fp16=False,
        dataloader_pin_memory=False,
    )

    # 4. Trainer
    dataset = load_from_disk(dataset_path)

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        # peft_config=peft_config,
        args=sft_config,
        processing_class=tokenizer,
    )

    # 5. Execute
    trainer.train()
    trainer.save_model(output_model_dir)

### Validation
def validate_script():
    """
    """
    import torch
    import json
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_path = output_sft_model_dir # Path to your SFT model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )

    # 1. Define the tools for the test
    tools = [{
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get current stock price",
            "parameters": {
                "type": "object",
                "properties": {"symbol": {"type": "string"}},
                "required": ["symbol"]
            }
        }
    }]

    # 2. Prepare the prompt using Qwen3's template
    messages = [
        {"role": "system", "content": "You are a helpful assistant with tool-calling capabilities."},
        {"role": "user", "content": "How much is Nvidia stock right now?"}
    ]

    # Qwen3 uses 'tools' kwarg in apply_chat_template to insert the XML blocks automatically
    text = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True # This enables the <think> block
    )

    # 3. Generate
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **inputs, 
        max_new_tokens=512, 
        temperature=0.6, # Recommended for Qwen3 thinking mode
        top_p=0.95
    )

    response = tokenizer.batch_decode(generated_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]

    print("--- MODEL OUTPUT ---")
    print(response)    

def main():

    prepare_sample_dataset_qwen()

    load_download_dataset()

    sft_trainer(convert_dataset_path, restore_model_dir_cache)

    # validate_script()

if __name__ == '__main__':
    main()

