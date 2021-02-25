MODEL_DIR="$(dirname $(readlink -f $0))/dall_e/models/"
wget "https://cdn.openai.com/dall-e/encoder.pkl" -P $MODEL_DIR
wget "https://cdn.openai.com/dall-e/decoder.pkl" -P $MODEL_DIR
