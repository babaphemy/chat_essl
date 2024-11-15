"""Copyright 2024 Everlasting Systems and Solutions LLC (www.myeverlasting.net).
All Rights Reserved.

No part of this software or any of its contents may be reproduced, copied, modified or adapted, without the prior written consent of the author, unless otherwise indicated for stand-alone materials.

For permission requests, write to the publisher at the email address below:
office@myeverlasting.net

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""
from trl import SFTTrainer,SFTConfig
from peft import LoraConfig
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, pipeline, training_args)
import logging
logger = logging.getLogger(__name__)

# loading model
llama_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path="aboonaji/llama2finetune-v3", 
                                                   quantization_config=BitsAndBytesConfig(
    load_in_8bit=True, 
    # bnb_4bit_compute_dtype=getattr(torch, "float16"),
    # bnb_4bit_quant_type="nf4"
))
llama_model.config.use_cache = False
llama_model.config.pretraining_tp = 1

# loading tokenizer
llama_tokenizer=AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path="aboonaji/llama2finetune-v3",
    trust_remote_code=True
)
llama_tokenizer.pad_token = llama_tokenizer.eos_token
llama_tokenizer.padding_side = "right"

#Training arguments
# training_args = TrainingArguments(
#     output_dir="./results",
#     per_device_train_batch_size=1,
#     max_steps=100,
#     # overwrite_output_dir=True,
#     # num_train_epochs=1,

#     # save_steps=10_000,
#     # save_total_limit=2,
#     # logging_steps=100,
#     # logging_dir="./logs",
#     # report_to="none",
#     # disable_tqdm=True
# )
training_args = SFTConfig(
    output_dir="./results",
    per_device_train_batch_size=1,
    max_steps=100,
    # overwrite_output_dir=True,
    # num_train_epochs=1,

    # save_steps=10_000,
    # save_total_limit=2,
    # logging_steps=100,
    # logging_dir="./logs",
    # report_to="none",
    # disable_tqdm=True
)
dataset = load_dataset("aboonaji/wiki_medical_terms_llam2_format", split="train")
sft_trainer = SFTTrainer(
    model=llama_model,
    args=training_args,
    train_dataset=dataset, 
    peft_config=LoraConfig(
        task_type="CAUSAL_LLM",
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
    ),
    
)
# Training the model
sft_trainer.train()

# chat 
user_prompt = "what is babeosis?"
text_gen_pipeline = pipeline(task="text-generation", model=llama_model, tokenizer=llama_tokenizer, max_length=300)
model_answer = text_gen_pipeline(f"<s>[INST] {user_prompt} [/INST]")
print(model_answer[0]['generated_text'])
