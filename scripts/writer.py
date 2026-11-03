import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parent.parent
WRITER = "wesjos/Qwen3.5-4B-Fuck"
NEVER = re.compile(r"n[i1!]gg|f[a@]gg?[o0]t|\bf[a@]g\b|retard|tr[a@]nn|ch[i1]nk|\bsp[i1]c\b|k[i1]ke|dyke|"
                   r"g[o0][o0]k|wetback|beaner|kill yourself|suicid|\bkys\b", re.I)
GROUPS = re.compile(r"\b(gay|lesbian|homo|queer|trans|black|white|asian|chinese|mexican|indian|arab|jew|muslim|"
                    r"christian|african|latin|immigrant|autis|down syndrome|disabled|rape)\w*", re.I)
TIKTOK = ROOT / "data" / "style" / "tiktok_roasts.json"


def q4(name):
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    net = AutoModelForCausalLM.from_pretrained(
        name, quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                     bnb_4bit_compute_dtype=torch.bfloat16),
        device_map="cuda:0", low_cpu_mem_usage=True).eval()
    return tok, net


def chat(tok, system, user):
    m = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    q = tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    if tok.name_or_path == WRITER and not q.rstrip().endswith("</think>"):
        q += "<think>\n\n</think>\n\n"
    return q
