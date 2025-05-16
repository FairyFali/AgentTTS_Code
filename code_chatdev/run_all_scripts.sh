#!/bin/bash

#i=1
#for script in ./SRDD/data/data_ChatDev_format_test*.sh; do
#    echo "Running $script..."
#    echo "Processing $i"
#    ./"$script"
#    echo "Finished $script"
#    ((i++))
#done


rm -rf WareHouse/*

python3 run.py --config test_70b70b70b --name 'Mystic_Maze' --task 'Mystic Maze is a 3D action game where players navigate through a maze filled with mystical creatures, obstacles, and puzzles. Players can choose from different characters with unique abilities and must use their skills to overcome challenges and reach the end of the maze. Along the way, they can collect power-ups and weapons to aid them in their quest. With immersive graphics and engaging gameplay, Mystic Maze offers an exciting and thrilling action gaming experience.' --org 'SRDD_Action_Game'
python3 run.py --config test_8b70b70b --name 'Mystic_Maze' --task 'Mystic Maze is a 3D action game where players navigate through a maze filled with mystical creatures, obstacles, and puzzles. Players can choose from different characters with unique abilities and must use their skills to overcome challenges and reach the end of the maze. Along the way, they can collect power-ups and weapons to aid them in their quest. With immersive graphics and engaging gameplay, Mystic Maze offers an exciting and thrilling action gaming experience.' --org 'SRDD_Action_Game'
python3 run.py --config test_3b70b70b --name 'Mystic_Maze' --task 'Mystic Maze is a 3D action game where players navigate through a maze filled with mystical creatures, obstacles, and puzzles. Players can choose from different characters with unique abilities and must use their skills to overcome challenges and reach the end of the maze. Along the way, they can collect power-ups and weapons to aid them in their quest. With immersive graphics and engaging gameplay, Mystic Maze offers an exciting and thrilling action gaming experience.' --org 'SRDD_Action_Game'


python3 run.py --config test_70b8b70b --name 'Mystic_Maze' --task 'Mystic Maze is a 3D action game where players navigate through a maze filled with mystical creatures, obstacles, and puzzles. Players can choose from different characters with unique abilities and must use their skills to overcome challenges and reach the end of the maze. Along the way, they can collect power-ups and weapons to aid them in their quest. With immersive graphics and engaging gameplay, Mystic Maze offers an exciting and thrilling action gaming experience.' --org 'SRDD_Action_Game'
python3 run.py --config test_70b3b70b --name 'Mystic_Maze' --task 'Mystic Maze is a 3D action game where players navigate through a maze filled with mystical creatures, obstacles, and puzzles. Players can choose from different characters with unique abilities and must use their skills to overcome challenges and reach the end of the maze. Along the way, they can collect power-ups and weapons to aid them in their quest. With immersive graphics and engaging gameplay, Mystic Maze offers an exciting and thrilling action gaming experience.' --org 'SRDD_Action_Game'

python3 run.py --config test_70b70b8b --name 'Mystic_Maze' --task 'Mystic Maze is a 3D action game where players navigate through a maze filled with mystical creatures, obstacles, and puzzles. Players can choose from different characters with unique abilities and must use their skills to overcome challenges and reach the end of the maze. Along the way, they can collect power-ups and weapons to aid them in their quest. With immersive graphics and engaging gameplay, Mystic Maze offers an exciting and thrilling action gaming experience.' --org 'SRDD_Action_Game'
python3 run.py --config test_70b70b3b --name 'Mystic_Maze' --task 'Mystic Maze is a 3D action game where players navigate through a maze filled with mystical creatures, obstacles, and puzzles. Players can choose from different characters with unique abilities and must use their skills to overcome challenges and reach the end of the maze. Along the way, they can collect power-ups and weapons to aid them in their quest. With immersive graphics and engaging gameplay, Mystic Maze offers an exciting and thrilling action gaming experience.' --org 'SRDD_Action_Game'

python3 run.py --config base_config --coding_model "llama3b" --coding_sampling 1 --review_model "llama3b" --review_sampling 1 --test_model "llama3b" --test_sampling 1 --org 'SRDD_Action_Game' --examples 2


python3 chatdev/eval_quality.py --config test_xb70b70b


