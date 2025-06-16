```bash 
nohup python main_std2.py --train data/train.txt --test data/test.txt --output results/Predictions.txt --stats results/TrainingStats.txt --min_rating 0 --max_rating 100 --lr 0.0005 --reg 0.1 --grad_clip 100 --factors 40 --epochs 100 > train.log 2>&1 
``` 

```bash 
Start-Process -FilePath "python" `
  -ArgumentList "main_std2.py --train data/train.txt --test data/test.txt --output results/Predictions.txt --stats results/TrainingStats.txt --min_rating 0 --max_rating 100 --lr 0.0005 --reg 0.1 --grad_clip 100 --factors 40 --epochs 100" `
  -RedirectStandardOutput "train.log" `
  -RedirectStandardError "train_bar.log" `
  -WindowStyle Hidden 
``` 

```bash 
python main.py  --train data/train.txt --test data/test.txt  --output results/Predictions.txt --stats results/TrainingStats.txt --min_rating 0  --max_rating 100 --lr 0.0005 --reg 0.1 --grad_clip 100 --factors 40 --epochs 10 --model BPRMF 
``` 