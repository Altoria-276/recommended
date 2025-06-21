nohup python main.py \
  --train data/train.txt \
  --test data/test.txt \
  --output results/Predictions.txt \
  --stats results/TrainingStats.txt \
  --min_rating 0 \
  --max_rating 100 \
  --lr 0.0005 \
  --reg 0.1 \
  --grad_clip 100 \
  --factors 40 \
  --epochs 1 \
  --model NeuralMF 
  2>&1 > output.log &