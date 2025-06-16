# 超酷炫翻譯

一個多功能翻譯工具，支援螢幕截圖 OCR 翻譯和手動輸入翻譯，並可整合多種大型語言模型 (LLM) API。

## 主要功能

- **螢幕截圖翻譯**：
  - 使用者可自訂快捷鍵 (預設 `Ctrl+Alt+S`) 觸發螢幕選取。
  - 自由選取螢幕上的任何區域進行截圖。
  - 內建 OCR (光學字元辨識) 功能，自動從圖片中提取文字。
  - 支援多種 OCR 辨識語言 (例如：英文、日文、韓文、簡體中文、繁體中文等)。
  - 將辨識出的文字快速翻譯成使用者指定的目標語言。

- **手動輸入翻譯**：
  - 提供直觀的文字輸入框，方便使用者直接輸入或貼上需要翻譯的內容。
  - 即時翻譯輸入的文字到選定的目標語言。

- **多翻譯引擎支援**：
  - 預設使用 `translate` 函式庫進行快速翻譯。
  - 支援整合多個強大的 LLM API 作為進階翻譯引擎，包括：
    - OpenAI (例如 GPT-3.5, GPT-4)
    - DeepSeek
    - Ollama (支援本地部署的開源模型，如 Llama3)
    - Google Gemini
    - Groq
  - 使用者可以在設定中新增、編輯、刪除及選擇偏好的 LLM 翻譯引擎，並管理各自的 API 金鑰和模型 ID。

- **高度可自訂的設定**：
  - **快捷鍵設定**：自由更改觸發螢幕截圖的快捷鍵組合。
  - **語言設定**：分別為螢幕截圖翻譯和手動翻譯設定 OCR 辨識語言及目標翻譯語言。
  - **引擎管理**：輕鬆管理和切換不同的翻譯引擎。
  - **字數限制**：設定單次翻譯的最大字元數。
  - 所有設定將自動儲存於 `settings.json` 檔案中，方便下次使用。

- **系統匣圖示與操作**：
  - 應用程式啟動後會在系統匣顯示圖示，方便快速存取。
  - 右鍵點擊圖示可開啟設定介面或退出應用程式。

- **日誌記錄**：
  - 自動記錄應用程式的運行狀態和潛在錯誤到 `app.log` 和 `error.log` 檔案，便於問題排查。

## 執行已打包的應用程式 (Windows)

如果您不想從原始碼執行，可以下載已打包的應用程式：

1.  前往本專案的 GitHub Releases 頁面。
2.  下載最新版本的 `super_cool_translator_vx.x.x.zip` (其中 `x.x.x` 為版本號)。
3.  解壓縮下載的 `.zip` 檔案。
4.  進入解壓縮後的 `super_cool_translator` 資料夾。
5.  執行 `super_cool_translator.exe`。

首次執行後，建議依照下面的「設定」指引進行配置，特別是 API 金鑰。

## 開發與執行原始碼

1.  **安裝依賴**：
    本應用程式使用 PyTorch 進行 OCR。`requirements.txt` 中已包含 `torch` 和 `torchvision`。您可以直接安裝所有依賴：
        ```bash
        pip install -r requirements.txt
        ```
        此預設安裝將使用 CPU 版本的 PyTorch 進行 OCR。如果您的系統配備了 NVIDIA GPU 並希望嘗試 GPU 加速，您可以參考 PyTorch 官方網站 ([https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)) 的說明，自行安裝與您 CUDA 環境相容的 PyTorch 版本。但請注意，本應用程式預設配置為使用 CPU。
2.  **運行程式**：
    ```bash
    python main.py
    ```
3.  **設定 (首次使用建議)**：
    - 程式啟動後，右鍵點擊系統匣圖示，選擇「設定」。
    - **API 金鑰設定**：
        - 如果您希望使用 LLM (如 OpenAI, Gemini, DeepSeek, Groq) 進行翻譯，您必須設定對應的 API 金鑰。
        - 應用程式在 `settings.json` 中預設為這些服務提供了佔位符金鑰 (例如 `"YOUR_OPENAI_API_KEY_HERE"`)。
        - 請在「設定」視窗的「LLM 引擎管理」中，選擇您要使用的引擎，點擊「編輯」，然後輸入您自己的有效 API 金鑰和模型 ID (如果適用)。
        - **重要**：若使用非 `ollama` 引擎且 API 金鑰為空或仍為預設佔位符，翻譯請求將不會執行，並會在翻譯結果區顯示錯誤提示。
        - `ollama` 引擎為本地部署，不需要雲端 API 金鑰，但需確保 Ollama 服務正在本機運行且已下載對應模型。
    - **其他設定**：
        - 設定您偏好的快捷鍵 (預設 `Ctrl+Alt+S`)。
        - 設定 OCR 辨識語言 (用於截圖翻譯)。
        - 設定目標翻譯語言。

## 打包應用程式 (供開發者)

如果您修改了原始碼並希望重新打包成獨立的 Windows 執行檔，可以使用 PyInstaller。

1.  確保已安裝 PyInstaller：
    ```bash
    pip install pyinstaller
    ```
2.  在專案根目錄下執行以下指令：
    ```bash
    pyinstaller --name super_cool_translator --onedir --windowed --add-data "settings.json:." --add-data "icon.ico:." main.py
    ```
    - `--name super_cool_translator`: 指定輸出應用程式的名稱。
    - `--onedir`: 將所有依賴打包到一個資料夾中 (相對於 `--onefile` 的單一執行檔，`--onedir` 模式通常啟動更快且問題較少)。
    - `--windowed`: 執行時不顯示主控台 (命令提示字元) 視窗。
    - `--add-data "settings.json:."`: 將 `settings.json` 檔案包含到打包後的應用程式根目錄。
    - `--add-data "icon.ico:."`: 將 `icon.ico` 檔案包含到打包後的應用程式根目錄。
    - `main.py`: 您的主應用程式腳本。

3.  打包完成後，可執行的應用程式將位於 `dist/super_cool_translator` 資料夾內。相關的建置檔案會產生在 `build/` 資料夾，而 `.spec` 檔案則包含了 PyInstaller 的建置設定。

## 檔案結構

- `main.py`: 應用程式主程式碼。
- `requirements.txt`: Python 依賴套件列表。
- `settings.json`: 使用者設定檔。
- `icon.ico`: 應用程式圖示。
- `README.md`: 本說明文件。
- `.gitignore`: Git 版本控制忽略列表。
- `app.log`: 應用程式運行日誌 (自動生成)。
- `error.log`: 錯誤日誌 (自動生成)。
- `startup_crash.log`: 啟動失敗日誌 (自動生成)。
- `dist/`: 打包後應用程式的輸出目錄 (由 PyInstaller 生成)。
- `build/`: PyInstaller 建置過程中產生的暫存檔案目錄 (由 PyInstaller 生成)。
- `super_cool_translator.spec`: PyInstaller 設定檔 (由 PyInstaller 生成)。

## 注意事項

- 使用 LLM 翻譯引擎可能需要有效的 API 金鑰，並可能產生相應的費用。
- Ollama 引擎需要在本地成功部署並運行相應模型。
- 日誌檔案 (`.log`) 和虛擬環境目錄 (`.venv`) 建議加入到 `.gitignore` (若使用 Git)。
