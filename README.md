
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


### QuickStart: Running Function Call SFT on Open Source Agent Models

In this tutorial, we will run function call SFT datasets and open source LLM model.

We are using the `transformers` and `trl` package as the base lib for SFT.

Firstly, download data and put the json file in folder e.g. "download_dataset_path": ../dataset/function_call/example_deepnlp_agent_function_call_202510.json 

Download Full [Dataset Document](https://deepnlp.org/doc/ai_agent_datasets) and 50 examples function call [Direct Download](https://static.aiagenta2z.com/scripts/doc/file/06766e91894147319ffd0116b04ff94d/example_deepnlp_agent_function_call_202510.json)

```
mkdir ../train/dataset/function_call
cd ../train/dataset/function_call
wget https://static.aiagenta2z.com/scripts/doc/file/06766e91894147319ffd0116b04ff94d/example_deepnlp_agent_function_call_202510.json
```

Running SFT Function Call Datasets
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


### Related

[MCP Marketplace](https://www.deepnlp.org/store/ai-agent/mcp-server)   
[AI Agent Marketplace](https://www.deepnlp.org/store/ai-agent)   
[ChatGPT Apps Store Marketplace](https://www.deepnlp.org/store/chatgpt-app)   

