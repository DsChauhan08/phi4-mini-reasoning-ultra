!git clone https://github.com/ggerganov/llama.cpp
%cd llama.cpp
!pip install -U -r requirements.txt huggingface_hub tiktoken sentencepiece transformers
!cmake -B build
!cmake --build build --config Release -j --target llama-quantize llama-cli


!huggingface-cli download dschauhan08/phi4-mini-ultra-reasoning --local-dir model_hf --local-dir-use-symlinks False

# --- 3. APPLY TOKENIZER PATCH ---
!wget -q -O model_hf/tokenizer_config.json https://huggingface.co/microsoft/phi-4-mini-instruct/resolve/main/tokenizer_config.json
!wget -q -O model_hf/tokenizer.json https://huggingface.co/microsoft/phi-4-mini-instruct/resolve/main/tokenizer.json
!wget -q -O model_hf/special_tokens_map.json https://huggingface.co/microsoft/phi-4-mini-instruct/resolve/main/special_tokens_map.json
!wget -q -O model_hf/added_tokens.json https://huggingface.co/microsoft/phi-4-mini-instruct/resolve/main/added_tokens.json

# --- 4. EXPORT TO F16 GGUF ---
!python convert_hf_to_gguf.py model_hf --outfile phi4-mini-ultra-reasoning-f16.gguf --outtype f16

# --- 5. 🚨 CRITICAL: FREE UP DISK SPACE 🚨 ---
# We no longer need the original safetensors. Deleting this frees up 15.5 GB of space!
!rm -rf model_hf

# --- 6. RUN QUANTIZATIONS ---
# Now we have plenty of disk space to generate the gigabyte files
!./build/bin/llama-quantize phi4-mini-ultra-reasoning-f16.gguf phi4-mini-ultra-reasoning-q8_0.gguf q8_0
!./build/bin/llama-quantize phi4-mini-ultra-reasoning-f16.gguf phi4-mini-ultra-reasoning-q6_k.gguf q6_k
!./build/bin/llama-quantize phi4-mini-ultra-reasoning-f16.gguf phi4-mini-ultra-reasoning-q4_k_m.gguf q4_k_m
!./build/bin/llama-quantize phi4-mini-ultra-reasoning-f16.gguf phi4-mini-ultra-reasoning-q2_k.gguf q2_k

# --- 7. ISOLATE FILES FOR CLEAN UPLOAD ---
# Move ONLY our 5 newly generated models into a dedicated folder
!mkdir -p final_models
!mv *.gguf final_models/

# --- 8. UPLOAD TO HUGGING FACE ---
# Replace YOUR_NEW_TOKEN_HERE with a freshly generated write token
!HF_TOKEN="token" huggingface-cli upload dschauhan08/phi4-mini-ultra-reasoning-GGUF ./final_models .


%%bash
# 1. Move into the working directory
cd /content/llama.cpp

# 2. Loop through every GGUF file in our final folder
for model in final_models/*.gguf; do
    echo -e "\n\n"
    echo "==================================================================="
    echo "🧠 LOADING AND TESTING: $model"
    echo "==================================================================="
    
    # Run a quick reasoning test (generating 100 tokens to prove it works)
    ./build/bin/llama-cli \
      -m "$model" \
      -n 100 \
      -c 512 \
      -p "<|im_start|>user\nThink step-by-step: Which is heavier, 1 kilogram of feathers or 1 kilogram of steel?<|im_end|>\n<|im_start|>assistant\n<think>"
    
    echo -e "\n"
    echo "==================================================================="
    echo "🧹 TEST COMPLETE. UNLOADING $model FROM RAM..."
    echo "==================================================================="
    sleep 2 # Give the system 2 seconds to completely clear the memory
done

echo -e "\n🎉 ALL MODELS TESTED SUCCESSFULLY!"
