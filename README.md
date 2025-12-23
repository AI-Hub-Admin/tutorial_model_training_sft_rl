
### Tutorial on Running SFT/RL/Post Training of Function Calls and RL Dataset

[LLM & AI Agent Dataset Doc](https://deepnlp.org/doc/ai_agent_datasets)|[Agent Tool Use Web](https://agent.deepnlp.org/)|[AI Agent Marketplace](https://www.deepnlp.org/store/ai-agent) | [Huggingface](https://huggingface.co/datasets/DeepNLP/Agent-Function-Calling-Open-Dataset)

### Requirements
```
datasets>=4.4.0
torch>=2.2.0
transformers>=4.57.0
trl>=0.26.2
```

### Downloading Dataset 

[50 demo examples](https://deepnlp.org/store/dataset/dataset/pub-deepnlp/agent-function-calling-open-dataset-example)

[1K examples](https://www.deepnlp.org/store/dataset/dataset/pub-deepnlp/agent-function-calling-open-dataset)

Register at [DeepNLP Datasets](https://deepnlp.org/workspace/keys) and new users will have bonus credit at [Billing](https://deepnlp.org/workspace/billing) enought to download the datasets.

For more production ready commercial dataset, Please contact deepnlp.contact@gmail.com



### QuickStart: 

#### 1.Running Function Call SFT on Open Source Agent Models

In this tutorial, we will run function call SFT datasets and open source LLM model.

We are using the `transformers` and `trl` package as the base lib for SFT.

Firstly, download data and put the json file in folder e.g. "download_dataset_path": ../dataset/function_call/example_deepnlp_agent_function_call_202510.json 

Download Full [Dataset Document](https://deepnlp.org/doc/ai_agent_datasets) and 50 examples function call [Direct Download](https://static.aiagenta2z.com/scripts/doc/file/a456552a82d94f319c1d77e8ceb443c9/deepnlp_rl_feedback_202510.json)

[1K RL Function Call Example](https://www.deepnlp.org/store/dataset/dataset/pub-deepnlp/agent-reinforcement-learning-open-dataset)

```
mkdir ../train/dataset/function_call
cd ../train/dataset/function_call
wget https://static.aiagenta2z.com/scripts/doc/file/06766e91894147319ffd0116b04ff94d/example_deepnlp_agent_function_call_202510.json
```

```
cd ../../scripts
python train_function_call_sft_qwen3.py

```



Start SFT Output
```
In [54]:     trainer.train()
    ...:     trainer.save_model(output_model_dir)
    ...:

### Logs 
The tokenizer has new PAD/BOS/EOS tokens that differ from the model config and generation config. The model config and generation config were aligned accordingly, being updated with the tokenizer's values. Updated tokens: {'bos_token_id': None, 'pad_token_id': 151643}.
  0%|                                                                                                                                                  | 0/21 [00:00<?, ?it/s]/Library/Frameworks/
  num_items_in_batch = self.accelerator.gather(num_items_in_batch.to(device)).sum()

```

#### 2.Running Reinforcement Learning on RL Datasets

Download Full [Dataset Document](https://deepnlp.org/doc/ai_agent_datasets) and 50 examples Function Call RL Data Datasets[Direct Download]()


##### 2.1 Scalar Rewards on a List of Messages

This RL scenario is suitable to improve on a list of tool call and messages session list, not the tool_call output. 

Converted Dataset Example: 

prompt: is a list of messages, in which users asked Agent to draw of pie-chart of 30 categories.
rewards: 1.0, user clicks on the final responses, the MCP tool call and the returned the tool results can generated url <url>https://mdn.alipayobjects.com/one_clip/afts/img/JEeRRbKd44wAAAAAYzAAAAgAoEACAQFr/original</url>

Use the `scalar_lookup_reward_fn()` in the tutorial.

```
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

```


Preparing Datasets for trl

```
mkdir ../train/dataset/rl
cd ../train/dataset/rl
wget https://static.aiagenta2z.com/scripts/doc/file/4619bcee15bf4a189044052791c44eac/example_deepnlp_rl_feedback_202510.json
```

Running RL Training of the Datasets


**`trl` package**

```
cd ../../scripts
python train_rl_grpo_qwen3_openai_trl.py
```

**`verl` package**

```
cd ../../scripts

python train_rl_grpo_qwen3_openai_verl.py
## verl_train.parquet, "../train/dataset/rl/verl_train.parquet", Download Cache Model From Huggingface/Dashscope


python -m verl.trainer.main_ppo \
    data.train_files="../train/dataset/rl/verl_train.parquet" \
    data.val_files="../train/dataset/rl/verl_train.parquet" \
    data.train_batch_size=1 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path="../train/model/qwen3/Qwen3-0.6B/Qwen/Qwen3-0___6B" \
    actor_rollout_ref.actor.ppo_mini_batch_size=1 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.name=hf \
    trainer.n_gpus_per_node=0 \
    trainer.nnodes=1 \
    trainer.project_name="grpo_qwen3_rl_verl"

```


##### 2.2 Reward of Tool Call Ground Truth vs Current Completion

If you want to improve the tool_call accuracy Pass@K, such as 'tool choice' accuracy or parameters inference accuracy, you can organize the datasets together with the [function call datasets](https://deepnlp.org/store/ai-agent/ai-agent/pub-deepnlp/agent-function-calling-open-dataset) and get the completion history of available tools, prompt, and tool_call output.

```
	## Example Dataset Formating using tool_call_reward_fn reward function

	prompt Input: [{"role": "user", "content": "What is the weather in SF?"}]
	ground truth: get_weather(city='SF')
	reward function:
        {
            "data_source": "tool_call_weather",
            "prompt": [{"role": "user", "content": "What is the weather in SF?"}],
            "ability": "tool_call",
            "reward_model": {
                "style": "rule",
                "ground_truth": "get_weather(city='SF')"
            }
        }

    ## function call result in the function call dataset
    ## Completion Output
	{"tool_calls":[{"id":"call_41d68645f2c94f1db65dd5","type":"tool_use","function":{"name":"maps_geocode","arguments":"{\"address\": \"John F. Kennedy International Airport, New York\"}"}}]}
```

Use the `tool_call_reward_fn` in the tutorial, which fetch the tools_calls, and agaist the ground_truth if available.



### Related

[MCP Marketplace](https://www.deepnlp.org/store/ai-agent/mcp-server)   
[AI Agent Marketplace](https://www.deepnlp.org/store/ai-agent)   
[ChatGPT Apps Store Marketplace](https://www.deepnlp.org/store/chatgpt-app)   



