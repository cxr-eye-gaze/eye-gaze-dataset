# eye-gaze-ai
Use of eye-gaze and attention signals to bridge Human intuition and AI


### Run experiments in parallel
- Install [rush](https://github.com/shenwei356/rush) or [parallel](https://www.gnu.org/software/parallel/)
- Inside Models run:
    - for rush: ```rush -j7 --verbose '{} --gpus=$((({#})%8)) &> {#}.out' -i  experiment_temporal```
    - for parallel: ```parallel -j7 --verbose --colsep ' ' '{} --gpus=$((({%})%8)) &> {#}.out' < experiment_temporal```