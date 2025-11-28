@echo off
echo 🔥 Fire Simulation Pipeline Start...

:: 1. Training and Initial Evaluation
echo 1/3: Running train.py (Convergence setting: 0.0)
python train.py -s data\chair -m output\chair_noconv_eval --eval --lambda_converge 0.0 --converge_interval 0

IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] train.py failed. Exiting...
    goto :EOF
)

:: 2. Rendering (Visualization)
echo 2/3: Running render.py
python render.py -m output\chair_noconv_eval --eval --skip_train

IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] render.py failed. Exiting...
    goto :EOF
)

:: 3. Metrics Calculation
echo 3/3: Running metrics.py
python metrics.py -m output\chair_noconv_eval

IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] metrics.py failed.
)

echo ✅ Pipeline finished successfully!