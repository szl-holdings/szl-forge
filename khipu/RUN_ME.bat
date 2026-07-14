@echo off
setlocal
cd /d "%~dp0"
title Khipu BrainNavigator Forge
echo ================================================================
echo   Khipu BrainNavigator forge  --  folder: %CD%
echo ================================================================
echo.
if not exist "train_khipu.py" goto WRONGDIR
set HF_HUB_ENABLE_HF_TRANSFER=0
echo [1/6] Checking GPU torch...
python -c "import torch,sys; print('torch',torch.__version__,'CUDA',torch.cuda.is_available()); sys.exit(0 if torch.cuda.is_available() else 3)"
if not errorlevel 3 goto TORCH_OK
echo    CPU-only torch found -- installing the CUDA build (large download, be patient)...
python -m pip install "torch==2.10.0" "torchvision==0.25.0" --index-url https://download.pytorch.org/whl/cu128 --force-reinstall --no-deps
python -c "import torch; print('torch now',torch.__version__,'CUDA',torch.cuda.is_available())"
:TORCH_OK
echo.
echo [2/6] Owner key...
if exist "owner_pubkey.json" echo    owner_pubkey.json already present -- skipping keygen.
if not exist "owner_pubkey.json" python sign_receipt.py keygen
echo.
echo [3/6] Training (QLoRA Qwen2.5-1.5B). First run downloads ~2.8GB and takes a while...
python train_khipu.py
if errorlevel 1 goto FAILTRAIN
echo.
echo [4/6] Rebirth into Ollama (make sure 'ollama serve' is running in another window)...
powershell -ExecutionPolicy Bypass -File "%~dp0rebirth-khipu.ps1"
echo.
echo [5/6] Training-set sanity gate (aborts if undertrained)...
python sanity_gate.py
if errorlevel 1 goto FAILSANITY
echo.
echo [6/6] Held-out eval...
python eval_khipu.py
if errorlevel 1 goto FAILEVAL
echo.
echo ================================================================
echo   DONE -- send these THREE files to Alloy:
echo ================================================================
if exist owner_pubkey.json echo   [OK]      owner_pubkey.json
if not exist owner_pubkey.json echo   [MISSING] owner_pubkey.json
if exist training_receipt.signed.json echo   [OK]      training_receipt.signed.json
if not exist training_receipt.signed.json echo   [MISSING] training_receipt.signed.json
if exist eval_receipt.signed.json echo   [OK]      eval_receipt.signed.json
if not exist eval_receipt.signed.json echo   [MISSING] eval_receipt.signed.json
echo.
echo   They are in: %CD%
goto END
:WRONGDIR
echo ERROR: the forge scripts are not in this folder.
echo Put RUN_ME.bat inside ...\docs\forge\khipu (next to the .py files) and run it there.
goto END
:FAILTRAIN
echo TRAINING FAILED. Scroll up, copy the red error text, paste it to Alloy.
goto END
:FAILSANITY
echo SANITY GATE FAILED (undertrained). Scroll up, copy the whole [sanity] block, paste it to Alloy.
goto END
:FAILEVAL
echo EVAL FAILED. Scroll up, copy the red error text, paste it to Alloy.
goto END
:END
echo.
pause
