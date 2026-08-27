# Reference: https://github.com/nianticlabs/simplerecon/blob/main/data_scripts/7scenes_preprocessing.py
# The scripts provided here are for reference only. Please ensure you have obtained the necessary licenses from the original dataset providers before proceeding.

mkdir -p data/7scenes
cd data/7scenes
scenes=("chess" "fire" "heads" "office" "pumpkin" "redkitchen" "stairs")

for scene in "${scenes[@]}"; do
    echo "=============================== Downloading 7scenes Data: $scene ==============================="
    
    wget "http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/$scene.zip"
    unzip "$scene.zip"
    
    sequences=("$scene"/*)
    
    for file in "${sequences[@]}"; do
        if [[ "$file" == *.zip ]]; then
            echo "Unpacking $file"
            unzip "$file" -d "$scene"
            rm "$file"
        fi
    done
done

for scene in "${scenes[@]}"; do
    rm "$scene.zip"
done
cd ../..

# prepare the dataset for mv_recon evaluation
python datasets/preprocess/prepare_7scenes.py