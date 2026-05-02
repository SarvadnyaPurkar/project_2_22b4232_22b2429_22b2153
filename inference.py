import os
import gc
import re
import torch
import pandas as pd
from PIL import Image
import time
import argparse
from collections import Counter
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from transformers import AutoModelForCausalLM, AutoTokenizer



# os.environ["HF_HOME"] = "./weights/hf_home"
# os.environ["TRANSFORMERS_CACHE"] = "./weights/hf_home"
# os.environ["HF_DATASETS_CACHE"] = "./weights/hf_home"
base_dir = os.path.abspath("./weights/hf_home")

os.environ["HF_HOME"] = base_dir
os.environ["TRANSFORMERS_CACHE"] = base_dir
os.environ["HF_DATASETS_CACHE"] = base_dir
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
PROMPTS = [
    # 1️⃣ Standard reasoning
    "Read the MCQ carefully. Understand the concept and eliminate incorrect options. Output ONLY A, B, C, or D. If unsure, output 5.",

    # 2️⃣ Elimination-focused
    "Eliminate incorrect options first, then choose the best answer. Output ONLY A, B, C, or D. If uncertain, output 5.",

    # 3️⃣ Concept-first reasoning
    "Identify the deep learning concept internally and evaluate all options, but DO NOT explain. Output ONLY a single character: A, B, C, or D. If unsure, output 5.", 

    # 4️⃣ Strict format (low hallucination)
    "IMPORTANT: Output only one character: A, B, C, or D. Do not explain. If uncertain, output 5."

]
# =========================
# DEVICE + MODEL SELECTION
# =========================

# ===== CONFIG =====
model_id = "Qwen/Qwen2-VL-2B-Instruct"
save_path = "./weights/qwen2_vl_2b"
os.makedirs(save_path, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


print(f"Using Qwen2-VL-2B ({device})")
print(f"Using LLaMA-3-8B ({device})")
# print("Using Qwen2-VL-2B (GPU)")

# Existing Qwen2B setup
model_path = "./weights/qwen2_vl_2b"
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    low_cpu_mem_usage=True,
    local_files_only=True,
)
processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)

# NEW: LLaMA-3-8B reasoning model
llama_path = "./weights/llama3_8b"   # folder where you saved the model offline
llama_model = AutoModelForCausalLM.from_pretrained(
    llama_path,
    device_map="auto",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    local_files_only=True,
    low_cpu_mem_usage=True,
)
llama_tokenizer = AutoTokenizer.from_pretrained(llama_path, local_files_only=True)






def final_decision(answers):
    valid = [a for a in answers if a != 5]

    if len(valid) == 0:
        return 5

    counter = Counter(valid)
    best, count = counter.most_common(1)[0]

    # 🔥 Strong agreement
    if count >= 3:
        return best

    # 🔥 Moderate agreement
    if count == 2:
        if len(counter) <= 2:
            return best
        else:
            return 5

    return 5

def verify_answer(image_path, answer):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((512, 512))

    verify_prompt = f"""
    The selected answer is {answer}.
    Check if this is correct for the MCQ in the image.

    If the answer is correct, output YES.
    If the answer is incorrect, output NO.
    Do not explain.
    """

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": verify_prompt}
        ]
    }]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False, 
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.eos_token_id
        )

    result = processor.decode(
        output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip().upper()

    # cleanup
    del inputs, output
    torch.cuda.empty_cache()
    gc.collect()

    if "YES" in result:
        return True
    else:
        return False

def extract_answer(text):
    text = text.strip()

    # 1️⃣ Direct number
    if text in ["1", "2", "3", "4", "5"]:
        return int(text)

    # 2️⃣ Pattern like (A), (B), etc.
    match = re.findall(r'\(([A-D])\)', text.upper())
    if match:
        last = match[-1]
        return {"A":1, "B":2, "C":3, "D":4}[last]

    # 3️⃣ Pattern like "option C" or "answer is C"
    match = re.findall(r'\b([A-D])\b', text.upper())
    if match:
        last = match[-1]
        return {"A":1, "B":2, "C":3, "D":4}[last]

    # 4️⃣ Numbers in text
    matches = re.findall(r'\b[1-5]\b', text)
    if matches:
        return int(matches[-1])

    return 5

def predict_image(image_path):
    answers = []
    image = Image.open(image_path).convert("RGB").resize((512, 512))

    for prompt in PROMPTS:
        # Step 1: Use Qwen2B to extract MCQ text from image
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": "Extract the MCQ text only."}
            ]
        }]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)

        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        mcq_text = processor.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        
        # Step 2: Send extracted text + reasoning prompt to LLaMA
        reasoning_prompt = f"{prompt}\nMCQ:\n{mcq_text}"
        llama_inputs = llama_tokenizer(reasoning_prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            llama_output = llama_model.generate(**llama_inputs, max_new_tokens=20, do_sample=False)

        result = llama_tokenizer.decode(llama_output[0], skip_special_tokens=True)


        ans = extract_answer(result)
        print("RAW:", result, "| FINAL:", ans)
        answers.append(ans)

        # cleanup
        del inputs, output, llama_inputs, llama_output

        torch.cuda.empty_cache()
        gc.collect()

    best_answer = final_decision(answers)

    if best_answer != 5 and answers.count(best_answer) >= 3:
        is_correct = verify_answer(image_path, ["A","B","C","D"][best_answer-1])
        if not is_correct:
            return 5

    return best_answer



def run_inference(test_csv, image_dir, output_file="submission.csv"):
    df = pd.read_csv(test_csv)

    predictions = []

    for _, row in df.iterrows():
        image_name = row["image_name"]

        # ensure .png extension
        # if not image_name.endswith(".png"):
        #     image_name += ".png"
        valid_exts = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff")
        if not image_name.lower().endswith(valid_exts):
            # default to .png if no valid extension
            image_name += ".png"
        image_path = os.path.join(image_dir, image_name)

        print(f"\nProcessing: {image_name}")

        if not os.path.exists(image_path):
            print("Image missing → skipping")
            predictions.append(5)
            continue

        try:
            pred = predict_image(image_path)

        except Exception as e:
            print("Error:", e)
            pred = 5

        print("FINAL:", pred)

        predictions.append(pred)

        # cleanup (important for memory)
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    # create submission
    df["option"] = predictions
    df["id"] = df["image_name"]

    df[["id", "image_name", "option"]].to_csv(output_file, index=False)

    print("\nSubmission saved:", output_file)

def print_gpu_info():
    if torch.cuda.is_available():
        gpu_id = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(gpu_id)

        total_mem = torch.cuda.get_device_properties(gpu_id).total_memory / (1024**3)
        reserved = torch.cuda.memory_reserved(gpu_id) / (1024**3)
        allocated = torch.cuda.memory_allocated(gpu_id) / (1024**3)

        print("\n===== GPU INFO =====")
        print(f"GPU: {gpu_name}")
        print(f"Total Memory: {total_mem:.2f} GB")
        print(f"Reserved Memory: {reserved:.2f} GB")
        print(f"Allocated Memory: {allocated:.2f} GB")
        print(f"CUDA Version: {torch.version.cuda}")
        print("====================\n")
    else:
        print("\nRunning on CPU\n")

if __name__ == "__main__":
    start_time = time.time()

    print_gpu_info()
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", type=str, required=True)
    args = parser.parse_args()

    test_csv = os.path.join(args.test_dir, "test.csv")
    image_dir = os.path.join(args.test_dir, "images")

    run_inference(test_csv, image_dir)

    end_time = time.time()

    total_time = end_time - start_time

    print("\n===== EXECUTION TIME =====")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Total time: {total_time/60:.2f} minutes")
    print("==========================")
