python3 run.py --name 'Mystic_Maze' --task 'Mystic Maze is a 3D action game where players navigate through a maze filled with mystical creatures, obstacles, and puzzles. Players can choose from different characters with unique abilities and must use their skills to overcome challenges and reach the end of the maze. Along the way, they can collect power-ups and weapons to aid them in their quest. With immersive graphics and engaging gameplay, Mystic Maze offers an exciting and thrilling action gaming experience.' --org 'SRDD_Action_Game' --config test_8

python3 chatdev/eval_quality.py --config test_8
aws s3 cp --recursive ./logs/ s3://faliwang-data/ChatDev/logs/

# Specify the directory to clean up
TARGET_DIR="./WareHouse/"

# Check if the target directory exists
if [ -d "$TARGET_DIR" ]; then
    # Delete all files and folders within the target directory
    rm -rf "$TARGET_DIR"/*
    echo "All files and folders in $TARGET_DIR have been deleted."
else
    echo "Error: Target directory does not exist."
fi