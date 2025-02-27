import torch
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import TrainingArguments
from datasets import Dataset
import random
import numpy as np
from unsloth import is_bfloat16_supported, FastLanguageModel
from wtpsplit import SaT
import gc
import os


sat = SaT("sat-3l")
sat.half().to("cuda")
MODEL_NAME = "unsloth/gemma-2-9b-it-bnb-4bit"
PROMPT = "Correct the following text, making only minimal changes where necessary."
LEARNING_RATE = 5e-5
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 2
max_seq_length = 2560
LORA_RANK = 128
LORA_ALPHA = 64
char_limit = 1_000_000_000
max_tokens = 300
OUTPUT_DIR = "G500v3all"
PREDS_PATH = "G500v3allTest"
dir = f"Preds/{PREDS_PATH}"
if not os.path.exists(dir):
    os.makedirs(dir)

def md_to_dict(md):
    essay_dict = {}
    for essay in md.split("### essay_id = ")[1:]:
        (essay_id,text) = essay.split("\n", maxsplit=1)
        essay_dict[essay_id] = text.strip("\n")
        
    return essay_dict


def dict_to_md(essay_dict):
    md = ""
    for essay_id, essay_text in essay_dict.items():
        md += "### essay_id = {}\n{}\n\n".format(essay_id, essay_text)
    return md


def generate_in_batches(model, tokenizer, input_texts, last_prompt_part, batch_size=16, max_length=1536, max_new_tokens=1024, use_cache=True):
    all_outputs = []
    
    for i in range(0, len(input_texts), batch_size):
        batch_texts = input_texts[i : i + batch_size]
        input_ids = tokenizer(batch_texts, return_tensors="pt", padding="longest", max_length=max_length).to("cuda")
        output_sequences = model.generate(**input_ids, max_new_tokens=max_new_tokens, use_cache=use_cache)
        outputs = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
        all_outputs.extend(outputs)
    
    only_responses = []

    for output in all_outputs:
        idx = output.index(last_prompt_part)
        only_responses.append(output[idx+len(last_prompt_part):])
        
    return only_responses


def set_deterministic(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    
seed = 42
set_deterministic(seed=seed)


dataset_core_filenames = [
    "czech/NatForm/cs-natform",
    "czech/NatWebInf/cs-natwebinf",
    "czech/Romani/cs-romani",
    "czech/SecLearn/cs-seclearn",
    "english/writeandimprove2024/en-writeandimprove2024",
    "estonian/EIC/et-eic",
    "estonian/EKIL2/et-ekil2",
    "german/Merlin/de-merlin",
    "greek/GLCII/el-glcii",
    "icelandic/IceEC/is-IceEC",
    "icelandic/IceL2EC/is-IceL2EC",
    "italian/Merlin/it-merlin",
    "russian/rulec-gec/ru-rulec",
    "latvian/LaVA/lv-lava",
    "slovene/Solar-Eval/sl-solar_eval",
    "swedish/SweLL_gold/sv-swell_gold",
    "ukrainian/ua-gec/uk-ua_gec"
]

train_dataset_source_filenames = []
train_dataset_target_filenames = []

for name in dataset_core_filenames:
    for x in range(6):
        if os.path.isfile(f"{name}-ref{x}-train.md"):
            train_dataset_source_filenames.append(f"{name}-orig-train.md")
            train_dataset_target_filenames.append(f"{name}-ref{x}-train.md")

# you can choose whether you compute dev or test files
dev_dataset_source_filenames = [f"{name}-orig-test.md" for name in dataset_core_filenames]


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = LORA_RANK, 
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"], 
    lora_alpha = LORA_ALPHA,
    lora_dropout = 0.0, 
    bias = "none",    
    use_gradient_checkpointing = "unsloth", 
    random_state = 3407,
    use_rslora = False, 
    loftq_config = None,
)

source_dicts = []
target_dicts = []

for filename in train_dataset_source_filenames:
    with open(filename, "r") as file:
        data = file.read()
    
    source_dicts.append(md_to_dict(data))

for filename in train_dataset_target_filenames:
    with open(filename, "r") as file:
        data = file.read()
    
    target_dicts.append(md_to_dict(data))

source_texts = []
target_texts = []

for d1, d2 in zip(source_dicts, target_dicts):
    chars = 0
    for key1, key2 in zip(d1, d2):
        if chars > char_limit:
            break
        chars += len(d1[key1])
        source_inputs = tokenizer(d1[key1]).input_ids
        target_inputs = tokenizer(d2[key2]).input_ids

        if len(source_inputs) < max_tokens and len(target_inputs) < max_tokens:
            source_texts.append(d1[key1])
            target_texts.append(d2[key2])
        else:
            splitted_source_texts = d1[key1].split("\n")
            splitted_target_texts = d2[key2].split("\n")

            if len(splitted_source_texts) != len(splitted_target_texts):
                continue
            
            for p1, p2 in zip(splitted_source_texts, splitted_target_texts):
                p1_inputs = tokenizer(p1).input_ids
                p2_inputs = tokenizer(p2).input_ids
                if len(p1_inputs) < max_tokens and len(p2_inputs) < max_tokens:
                    source_texts.append(p1)
                    target_texts.append(p2)
                else:
                    sat_split_1 = sat.split(p1)
                    sat_split_2 = sat.split(p2)
                    if len(sat_split_1) != len(sat_split_2):
                        continue
                    
                    curr_s1 = ""
                    curr_s2 = ""
                    for s1, s2 in zip(sat_split_1, sat_split_2):
                        curr_s1 += s1.strip() + " "
                        curr_s2 += s2.strip() + " "

                        s1_inputs = tokenizer(curr_s1).input_ids
                        s2_inputs = tokenizer(curr_s2).input_ids
                        if len(s1_inputs) > max_tokens and len(s2_inputs) > max_tokens:
                            source_texts.append(curr_s1.strip())
                            target_texts.append(curr_s2.strip())
                            curr_s1 = ""
                            curr_s2 = ""

                    if len(curr_s1) > 0 and len(curr_s2) > 0:
                        source_texts.append(curr_s1.strip())
                        target_texts.append(curr_s2.strip())


correction_prompt = """{}

### Text to correct:
{}

### Corrected text:
{}"""

EOS_TOKEN = tokenizer.eos_token
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = correction_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts}


response_template = "### Corrected text:\n"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

output_dir = "/projects/"
FastLanguageModel.for_training(model)

source_dev_dicts = []

for filename in dev_dataset_source_filenames:
    with open(filename, "r") as file:
        data = file.read()
    
    source_dev_dicts.append(md_to_dict(data))


dev_dicts = []
for d in source_dev_dicts:
    dev_dicts.append({})
    for key in d:
        source_inputs = tokenizer(d[key]).input_ids

        if len(source_inputs) < max_tokens:
            dev_dicts[-1][key] = d[key]
        else:
            inputs = []
            splitted_texts = d[key].split("\n")

            for s in splitted_texts:
                source_inputs = tokenizer(s).input_ids
                if len(source_inputs) < max_tokens:
                    inputs.append(s)
                else:
                    sat_split = sat.split(s)
                    space_splitted_inputs = []

                    curr_input = ""
                    for sat_s in sat_split:
                        curr_input += sat_s.strip() + " "
                        curr_tokens = tokenizer(curr_input).input_ids
                        
                        if len(curr_tokens) > max_tokens:
                            space_splitted_inputs.append(curr_input.strip())
                            curr_input = ""

                    if len(curr_input) > 0:
                        space_splitted_inputs.append(curr_input)
                    inputs.append(space_splitted_inputs)
            
            dev_dicts[-1][key] = inputs


for x in range(2):
    ds = Dataset.from_dict({
    "instruction": [PROMPT for x in range(len(source_texts))],
    "input": source_texts,
    "output": target_texts
    })
    dataset = ds.map(formatting_prompts_func, batched = True)
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        data_collator = collator,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        packing = False,
        args = TrainingArguments(
            seed = 3407+x,
            save_strategy="no",
            output_dir = output_dir,
            per_device_train_batch_size = BATCH_SIZE,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            learning_rate = LEARNING_RATE,
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            optim = "adamw_8bit",
            warmup_steps = 40,
            num_train_epochs=1,
            gradient_accumulation_steps = GRADIENT_ACCUMULATION_STEPS,
            logging_steps=50
        ),
    )
    trainer_stats = trainer.train()

torch.cuda.empty_cache()
gc.collect()

save_path = f"/Models/{OUTPUT_DIR}"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)


FastLanguageModel.for_inference(model)

input_texts = []

for d in dev_dicts:
    for key in d:
        if type(d[key]) == str:
            input_texts.append(d[key])
        else:
            for paragraph in d[key]:
                if type(paragraph) == str:
                    input_texts.append(paragraph)
                else:
                    for sentence in paragraph:
                        input_texts.append(sentence)

input_texts = [correction_prompt.format(
            PROMPT, # instruction
            t, # input
            "", # output - leave this blank for generation!
            ) for t in input_texts]


o = generate_in_batches(model=model, tokenizer=tokenizer, input_texts=input_texts, last_prompt_part=response_template, batch_size=16, max_new_tokens=600)


aligned_preds = []
idx = 0
for d in dev_dicts:
    aligned_preds.append({})
    for key in d:
        if type(d[key]) == str:
            aligned_preds[-1][key] = o[idx]
            idx += 1
        else:
            pred = []
            for paragraph in d[key]:
                if type(paragraph) == str:
                    pred.append(o[idx])
                    idx += 1
                else:
                    splitted_pred = []
                    for sentence in paragraph:
                        splitted_pred.append(o[idx].strip())
                        idx += 1
                    
                    pred.append(" ".join(splitted_pred))
            
            aligned_preds[-1][key] = "\n".join(pred)


for idx, a in enumerate(aligned_preds):
    md_data = dict_to_md(a)

    with open(f"PredsAllDev/{PREDS_PATH}/{idx}.md", "w") as file:
        file.write(md_data)
