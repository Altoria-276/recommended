# 创建结果目录（如果不存在）
New-Item -ItemType Directory -Force -Path results | Out-Null

# 后台运行训练和预测任务
Start-Process -NoNewWindow -FilePath python -ArgumentList @(
    "main.py",
    "--train", "data/train.txt",
    "--test", "data/test.txt",
    "--output", "results/Predictions.txt",
    "--stats", "results/TrainingStats.txt",
    "--min_rating", "0",
    "--max_rating", "100",
    "--lr", "0.0005",
    "--reg", "0.1",
    "--grad_clip", "100",
    "--factors", "40",
    "--epochs", "1"
) -RedirectStandardOutput "output.log" -RedirectStandardError "error.log"

Write-Host "任务已在后台启动，日志输出到 output.log 和 error.log"