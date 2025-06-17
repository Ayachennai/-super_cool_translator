print("--- Script execution started ---")
import customtkinter as ctk
from tkinter import messagebox
import langdetect
import threading
import sys
import time
import os
import traceback
import json
import time
import traceback
from datetime import datetime
from pynput import keyboard
from pynput.keyboard import Key, KeyCode
from PIL import Image, ImageTk, ImageGrab, ImageDraw, ImageFont, ImageOps, ImageStat
import easyocr
import torch
import numpy as np
from translate import Translator as TransEngine
import google.generativeai as genai
import openai
from google.api_core import exceptions as google_exceptions
from langdetect import detect, LangDetectException
import pystray

def get_application_path():
    if getattr(sys, 'frozen', False): # PyInstaller 打包後的執行檔
        application_path = os.path.dirname(sys.executable)
    else: # 從 .py 原始碼執行
        application_path = os.path.dirname(os.path.abspath(__file__))
    return application_path

# --- 常數定義 ---
BASE_APP_PATH = get_application_path()

ICON_FILE = os.path.join(BASE_APP_PATH, 'icon.ico')

# 設定日誌
APP_LOG_PATH = os.path.join(BASE_APP_PATH, 'app.log')
ERROR_LOG_PATH = os.path.join(BASE_APP_PATH, 'error.log')

def log_message(message):
    """將訊息寫入日誌檔案"""
    # 使用全域日誌檔案路徑
    with open(APP_LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")

def log_error(e):
    """記錄錯誤到獨立的錯誤日誌檔案"""
    with open(ERROR_LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(f'\n\n--- {datetime.now()} ---\n')
        f.write(f'錯誤類型: {type(e).__name__}\n')
        f.write(f'錯誤訊息: {str(e)}\n')
        f.write('追蹤資訊:\n')
        traceback.print_exc(file=f)

# 支援的語言
LANGUAGES = {
    '自動偵測': 'auto',
    '繁體中文': 'zh-tw',
    '簡體中文': 'zh-cn',
    '英文': 'en',
    '日文': 'ja',
    '韓文': 'ko',
    '法文': 'fr',
    '德文': 'de',
    '西班牙文': 'es',
    '俄文': 'ru',
    '泰文': 'th',
    '越南文': 'vi',
}
# 反向查找，從 code 找 name
REVERSE_LANGUAGES = {v: k for k, v in LANGUAGES.items()}

# EasyOCR 語言代碼映射 (舊設定轉換用)
TESSERACT_TO_EASYOCR = {
    'kor': 'ko',
    'jpn': 'ja',
    'eng': 'en',
    'chi_sim': 'ch_sim',
    'chi_tra': 'ch_tra'
}

OCR_LANGUAGES = {
    '自動偵測': 'auto',
    '韓文': 'ko',
    '日文': 'ja',
    '英文': 'en',
    '簡體中文': 'ch_sim',
    '繁體中文': 'ch_tra',
    '法文': 'fr',
    '德文': 'de',
    '西班牙文': 'es',
    '俄文': 'ru',
    '葡萄牙文': 'pt'
}

# 預設設定
DEFAULT_SETTINGS = {
    'hotkey': ['ctrl', 'alt', 'q'],
    'screenshot_ocr_lang': 'ko',
    'screenshot_to_lang': 'en',
    'manual_to_lang': 'zh-tw',
    'max_length': 500,
    'llm_engines': [],
    'selected_engine': 'Default' # Name of the selected engine
}

# 設定檔案路徑
SETTINGS_FILE = os.path.join(BASE_APP_PATH, 'settings.json')

def load_settings():
    """載入設定，並處理舊格式遷移"""
    defaults = {
        'hotkey': ['ctrl', 'alt', 'q'],
        'screenshot_ocr_lang': 'ko',
        'screenshot_to_lang': 'en',
        'manual_to_lang': 'zh-tw',
        'max_length': 500,
        'llm_engines': [],
        'selected_engine': 'Default'
    }
    
    if not os.path.exists(SETTINGS_FILE):
        return defaults

    try:
        with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
            settings = json.load(f)
    except (json.JSONDecodeError, IOError):
        return defaults

    # --- 設定遷移邏輯 ---
    migrated = False
    if 'to_lang' in settings:
        settings['screenshot_to_lang'] = settings.get('to_lang', 'en')
        settings['manual_to_lang'] = settings.get('to_lang', 'zh-tw')
        del settings['to_lang']
        migrated = True

    if 'ocr_lang' in settings:
        settings['screenshot_ocr_lang'] = settings.get('ocr_lang', 'ko')
        del settings['ocr_lang']
        migrated = True
    
    if migrated:
        save_settings(settings) # 儲存遷移後的設定

    # 確保所有預設鍵都存在
    for key, value in defaults.items():
        settings.setdefault(key, value)
        
    settings.pop('translation_engine', None)
    settings.pop('from_lang', None) # from_lang 永久為 auto，不再需要儲存

    return settings

def save_settings(settings):
    """儲存設定到檔案"""
    with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
        json.dump(settings, f, ensure_ascii=False, indent=4)

def save_engines(engines):
    """儲存引擎列表到設定檔"""
    settings = load_settings()
    settings['engines'] = engines
    with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
        json.dump(settings, f, ensure_ascii=False, indent=4)

def load_engines():
    """從設定檔載入引擎列表，並確保提供者資訊和預設模型存在"""
    settings = load_settings()
    engines = settings.get('engines', {})
    
    needs_update = False

    # 1. 向後相容：為沒有 provider 的舊設定添加 provider
    for name, engine in engines.items():
        if 'provider' not in engine:
            needs_update = True
            model_id = engine.get('model_id', '').lower()
            if 'gpt' in model_id:
                engine['provider'] = 'openai'
            elif 'gemini' in model_id:
                engine['provider'] = 'google'
            else:
                engine['provider'] = 'openai'

    # 2. 新增預設的 DeepSeek, Ollama 和 Groq 引擎 (如果不存在)
    default_engines = {
        "DeepSeek": {
            "provider": "deepseek",
            "api_key": "YOUR_DEEPSEEK_API_KEY",  # 提示用戶替換
            "model_id": "deepseek-chat"
        },
        "Ollama (Llama3)": {
            "provider": "ollama",
            "api_key": "",  # Ollama 不需要金鑰
            "model_id": "llama3"
        },
        "Groq (Mixtral)": {
            "provider": "groq",
            "api_key": "YOUR_GROQ_API_KEY",
            "model_id": "mixtral-8x7b-32768"
        }
    }

    for name, default_engine in default_engines.items():
        if name not in engines:
            engines[name] = default_engine
            needs_update = True
            print(f"新增預設引擎: {name}")

    # 如果設定已更新或新增，則儲存回檔案
    if needs_update:
        print("偵測到舊版或缺少預設引擎，正在自動更新設定檔...")
        save_engines(engines)
        
    return engines

def translate_with_llm(api_key, model_id, provider, text, from_lang, to_lang):
    """根據提供者使用對應的 LLM API 翻譯文字
    
    參數:
        api_key (str): 提供者的 API 金鑰
        model_id (str): 模型 ID
        provider (str): 提供者名稱 (openai, deepseek, ollama, google)
        text (str): 要翻譯的文字
        from_lang (str): 來源語言代碼
        to_lang (str): 目標語言代碼
    
    返回:
        str: 翻譯後的文字或錯誤訊息
    """
    from_lang_name = REVERSE_LANGUAGES.get(from_lang, from_lang)
    to_lang_name = REVERSE_LANGUAGES.get(to_lang, to_lang)
    
    # 準備翻譯提示
    system_prompt = """你是一個專業的翻譯員，專注於將文字翻譯成繁體中文。
    請只回傳翻譯後的文字，不要包含任何額外的解釋或註釋。
    請確保使用正確的繁體中文字符。"""
    
    user_prompt = f"""請將以下文字從 {from_lang_name} 翻譯成繁體中文。
    請確保使用正確的繁體中文字符，不要簡化。
    
    原文：
    {text}
    
    翻譯（繁體中文）："""
    
    try:
        if provider == 'openai':
            # --- 使用 OpenAI API ---
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3  # 較低的溫度以獲得更加一致的翻譯
            )
            if response.choices and response.choices[0].message.content:
                return response.choices[0].message.content.strip()
            return "OpenAI API 返回了空的回應。"
            
        elif provider == 'deepseek':
            # --- 使用 DeepSeek API ---
            # 注意：DeepSeek 的 Python SDK 與 OpenAI SDK 非常相似
            client = openai.OpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com/v1"
            )
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            if response.choices and response.choices[0].message.content:
                return response.choices[0].message.content.strip()
            return "DeepSeek API 返回了空的回應。"
            
        elif provider == 'ollama':
            # --- 使用 Ollama 本地模型 ---
            import ollama
            response = ollama.chat(
                model=model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                options={"temperature": 0.3}
            )
            return response['message']['content'].strip()
            
        elif provider == 'google':
            # --- 使用 Google Gemini API ---
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(model_id)
            prompt = f"""請將以下文字從 {from_lang_name} 翻譯成繁體中文。
            請確保使用正確的繁體中文字符，不要簡化。
            
            原文：
            {text}
            
            翻譯（繁體中文）："""
            response = model.generate_content(prompt)
            if response.text:
                return response.text.strip()
            return "Google API 返回了空的回應，內容可能被安全策略阻擋。"

        elif provider == 'groq':
            # --- 使用 GroqCloud API ---
            # Groq API 與 OpenAI API 格式相容
            # 您需要在 https://console.groq.com/keys 獲取 API 金鑰
            if not api_key:
                return "錯誤: Groq API 金鑰未設定。"
            
            client = openai.OpenAI(
                api_key=api_key,
                base_url="https://api.groq.com/openai/v1" # Groq 的 OpenAI 相容端點
            )
            response = client.chat.completions.create(
                model=model_id, # 例如: mixtral-8x7b-32768, llama3-8b-8192
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3
            )
            if response.choices and response.choices[0].message.content:
                return response.choices[0].message.content.strip()
            return "Groq API 返回了空的回應。"
            
        else:
            return f"不支援的模型提供者: {provider}"
            
    except ImportError as e:
        missing_lib = str(e).split()[-1].replace("'", "")
        return f"錯誤: 缺少必要的函式庫。請執行 'pip install {missing_lib}'。"
    except openai.AuthenticationError:
        return "OpenAI API 金鑰錯誤或權限不足。"
    except google_exceptions.PermissionDenied as e:
        return f"Google API 金鑰錯誤或權限不足: {e.message}"
    except google_exceptions.GoogleAPICallError as e:
        return f"Google API 呼叫錯誤: {e.message}"
    except Exception as e:
        return f"執行 {provider} 翻譯時發生未預期錯誤: {type(e).__name__}: {str(e)}"

class EngineEditWindow(ctk.CTkToplevel):
    def __init__(self, master, engine_data=None, on_save=None):
        super().__init__(master)
        self.title("新增/編輯 LLM 引擎")
        self.geometry("400x400")
        self.transient(master)
        self.grab_set()

        self.engine_data = engine_data or {}
        self.on_save = on_save

        # 引擎名稱
        ctk.CTkLabel(self, text="引擎名稱:").pack(pady=(10, 0))
        self.name_entry = ctk.CTkEntry(self, placeholder_text="例如: My-GPT-4")
        self.name_entry.pack(pady=5, padx=20, fill="x")
        self.name_entry.insert(0, self.engine_data.get('name', ''))

        # 模型提供者
        ctk.CTkLabel(self, text="模型提供者:").pack(pady=(10, 0))
        self.provider_var = ctk.StringVar(value=self.engine_data.get('provider', 'openai'))
        self.provider_menu = ctk.CTkOptionMenu(
            self, 
            variable=self.provider_var,
            values=['openai', 'deepseek', 'ollama', 'google', 'groq']
        )
        self.provider_menu.pack(pady=5, padx=20, fill="x")

        # API Key (Ollama 可能不需要)
        ctk.CTkLabel(self, text="API Key (Ollama 可留空):").pack(pady=(10, 0))
        self.api_key_entry = ctk.CTkEntry(self, placeholder_text="在此輸入 API Key", show="*")
        self.api_key_entry.pack(pady=5, padx=20, fill="x")
        self.api_key_entry.insert(0, self.engine_data.get('api_key', ''))

        # 模型 ID
        ctk.CTkLabel(self, text="模型 ID:").pack(pady=(10, 0))
        self.model_id_entry = ctk.CTkEntry(self, placeholder_text="例如: gpt-4-1106-preview")
        self.model_id_entry.pack(pady=5, padx=20, fill="x")
        self.model_id_entry.insert(0, self.engine_data.get('model_id', ''))
        
        # 根據提供者設置預設模型 ID
        self.set_default_model_id()
        self.provider_menu.configure(command=lambda _: self.set_default_model_id())

        # 保存按鈕
        self.save_btn = ctk.CTkButton(self, text="儲存", command=self.save)
        self.save_btn.pack(pady=20)
    
    def set_default_model_id(self):
        """根據選擇的提供者設置預設模型 ID"""
        provider = self.provider_var.get()
        defaults = {
            'openai': 'gpt-3.5-turbo',
            'deepseek': 'deepseek-chat',
            'ollama': 'llama3',  # 假設用戶已經下載了 llama3 模型
            'google': 'gemini-pro',
            'groq': 'mixtral-8x7b-32768'
        }
        
        current_text = self.model_id_entry.get()
        if not current_text or current_text in defaults.values():
            self.model_id_entry.delete(0, 'end')
            self.model_id_entry.insert(0, defaults.get(provider, ''))

    def save(self):
        name = self.name_entry.get().strip()
        provider = self.provider_var.get()
        api_key = self.api_key_entry.get().strip()
        model_id = self.model_id_entry.get().strip()

        if not name or not model_id:
            print("錯誤: 請填寫所有必填欄位。")
            return
            
        if provider != 'ollama' and not api_key:
            print("錯誤: 此提供者需要 API 金鑰。")
            return

        if self.on_save:
            self.on_save({
                'name': name,
                'provider': provider,
                'api_key': api_key,
                'model_id': model_id
            })
        self.destroy()

from tkinter import filedialog
import platform
if platform.system() == "Windows":
    import ctypes
    try:
        # Attempt to load user32 to ensure it's available early if needed for DPI awareness logic
        # This also helps to confirm ctypes is working as expected.
        ctypes.windll.user32 
    except AttributeError:
        # This might happen in some restricted environments or non-standard Python builds.
        # Log this occurrence if logging is available here, or handle appropriately.
        print("Warning: ctypes.windll.user32 not available, multi-monitor support might be affected.")
        # Set a flag or handle so later code doesn't assume user32 is always present
        # For now, we'll let it proceed and fail later if GetSystemMetrics is called without user32.
        pass

class SettingsWindow(ctk.CTkToplevel):
    def __init__(self, master, settings, on_save_callback):
        super().__init__(master)
        self.title("設定")
        self.geometry("480x700") # Increased height for new settings
        self.transient(master)
        self.grab_set()

        self.settings = settings
        self.on_save_callback = on_save_callback

        self.scrollable_frame = ctk.CTkScrollableFrame(self)
        self.scrollable_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # --- Screenshot Translation Settings ---
        ctk.CTkLabel(self.scrollable_frame, text="截圖翻譯設定", font=("Arial", 16, "bold")).pack(pady=(10, 5), anchor="w", padx=10)

        ctk.CTkLabel(self.scrollable_frame, text="OCR 辨識語言:").pack(pady=(5, 0), anchor="w", padx=10)
        current_ocr_lang_code = self.settings.get('screenshot_ocr_lang', 'ko')
        current_ocr_lang_name = next((name for name, code in OCR_LANGUAGES.items() if code == current_ocr_lang_code), '韓文')
        self.screenshot_ocr_lang_var = ctk.StringVar(value=current_ocr_lang_name)
        self.screenshot_ocr_lang_menu = ctk.CTkOptionMenu(self.scrollable_frame, 
                                                       values=list(OCR_LANGUAGES.keys()),
                                                       variable=self.screenshot_ocr_lang_var)
        self.screenshot_ocr_lang_menu.pack(pady=5, padx=10, fill="x")

        ctk.CTkLabel(self.scrollable_frame, text="目標語言:").pack(pady=(10, 0), anchor="w", padx=10)
        self.screenshot_to_lang_combo = ctk.CTkComboBox(self.scrollable_frame, values=list(LANGUAGES.keys()))
        self.screenshot_to_lang_combo.pack(pady=5, padx=10, fill="x")
        screenshot_to_lang_key = REVERSE_LANGUAGES.get(self.settings.get('screenshot_to_lang', 'en'), '英文')
        self.screenshot_to_lang_combo.set(screenshot_to_lang_key)

        # --- Manual Translation Settings ---
        ctk.CTkLabel(self.scrollable_frame, text="手動翻譯設定", font=("Arial", 16, "bold")).pack(pady=(20, 5), anchor="w", padx=10)

        ctk.CTkLabel(self.scrollable_frame, text="目標語言:").pack(pady=(5, 0), anchor="w", padx=10)
        self.manual_to_lang_combo = ctk.CTkComboBox(self.scrollable_frame, values=list(LANGUAGES.keys()))
        self.manual_to_lang_combo.pack(pady=5, padx=10, fill="x")
        manual_to_lang_key = REVERSE_LANGUAGES.get(self.settings.get('manual_to_lang', 'zh-tw'), '繁體中文')
        self.manual_to_lang_combo.set(manual_to_lang_key)

        # --- Translation Engine Settings ---
        ctk.CTkLabel(self.scrollable_frame, text="翻譯引擎設定", font=("Arial", 16, "bold")).pack(pady=(20, 5), anchor="w", padx=10)

        ctk.CTkLabel(self.scrollable_frame, text="預設翻譯引擎:").pack(pady=(5, 0), anchor="w", padx=10)
        engine_names = ["Default"] + [engine['name'] for engine in self.settings.get('llm_engines', [])]
        self.engine_combo = ctk.CTkComboBox(self.scrollable_frame, values=engine_names)
        self.engine_combo.pack(pady=5, padx=10, fill="x")
        self.engine_combo.set(self.settings.get('selected_engine', 'Default'))

        ctk.CTkLabel(self.scrollable_frame, text="LLM 引擎管理:").pack(pady=(15, 5), anchor="w", padx=10)
        self.engine_frame = ctk.CTkFrame(self.scrollable_frame, fg_color="transparent")
        self.engine_frame.pack(pady=5, padx=10, fill="x", expand=True)
        self.populate_engine_list()

        self.add_engine_btn = ctk.CTkButton(self.scrollable_frame, text="新增引擎", command=self.add_or_edit_engine)
        self.add_engine_btn.pack(pady=10, padx=10, anchor="e")

        # --- General Settings ---
        ctk.CTkLabel(self.scrollable_frame, text="常規設定", font=("Arial", 16, "bold")).pack(pady=(15, 5), anchor="w", padx=10)

        ctk.CTkLabel(self.scrollable_frame, text="設定截圖快捷鍵:").pack(anchor="w", padx=10)
        self.hotkey_entry = ctk.CTkEntry(self.scrollable_frame, placeholder_text="例如: ctrl+alt+q")
        self.hotkey_entry.pack(pady=5, padx=10, fill="x")
        self.hotkey_entry.insert(0, '+'.join(self.settings.get('hotkey', [])))
        
        ctk.CTkLabel(self.scrollable_frame, text="翻譯最大字元數:").pack(anchor="w", padx=10)
        self.max_length_entry = ctk.CTkEntry(self.scrollable_frame, placeholder_text="例如: 500")
        self.max_length_entry.pack(pady=5, padx=10, fill="x")
        self.max_length_entry.insert(0, str(self.settings.get('max_length', 500)))

        self.save_btn = ctk.CTkButton(self, text="儲存並關閉", command=self.save)
        self.save_btn.pack(pady=15, padx=10)

    def populate_engine_list(self):
        for widget in self.engine_frame.winfo_children():
            widget.destroy()
        engines = self.settings.get('llm_engines', [])
        if not engines:
            ctk.CTkLabel(self.engine_frame, text="尚未設定 LLM 引擎").pack()
            return
        for i, engine in enumerate(engines):
            item_frame = ctk.CTkFrame(self.engine_frame, fg_color=("gray80", "gray20"))
            item_frame.pack(fill="x", pady=2)
            ctk.CTkLabel(item_frame, text=engine.get('name', '未命名')).pack(side="left", padx=10)
            delete_btn = ctk.CTkButton(item_frame, text="刪除", width=60, command=lambda i=i: self.delete_engine(i))
            delete_btn.pack(side="right", padx=5, pady=5)
            edit_btn = ctk.CTkButton(item_frame, text="編輯", width=60, command=lambda i=i: self.add_or_edit_engine(i))
            edit_btn.pack(side="right", padx=5, pady=5)

    def add_or_edit_engine(self, index=None):
        engine_data = self.settings['llm_engines'][index] if index is not None else None
        def on_save_callback(new_data):
            if index is not None:
                self.settings['llm_engines'][index] = new_data
            else:
                self.settings['llm_engines'].append(new_data)
            self.populate_engine_list()
            engine_names = ["Default"] + [e['name'] for e in self.settings.get('llm_engines', [])]
            self.engine_combo.configure(values=engine_names)
        EngineEditWindow(self, engine_data=engine_data, on_save=on_save_callback)

    def delete_engine(self, index):
        self.settings['llm_engines'].pop(index)
        self.populate_engine_list()
        engine_names = ["Default"] + [e['name'] for e in self.settings.get('llm_engines', [])]
        self.engine_combo.configure(values=engine_names)
        if self.engine_combo.get() not in engine_names:
            self.engine_combo.set("Default")

    def save(self):
        """儲存所有設定並關閉視窗"""
        # 1. 儲存截圖翻譯設定
        ocr_lang_display = self.screenshot_ocr_lang_var.get()
        self.settings['screenshot_ocr_lang'] = OCR_LANGUAGES.get(ocr_lang_display, 'ko')
        to_lang_display = self.screenshot_to_lang_combo.get()
        self.settings['screenshot_to_lang'] = LANGUAGES.get(to_lang_display, 'en')

        # 2. 儲存手動翻譯設定
        manual_to_lang_display = self.manual_to_lang_combo.get()
        self.settings['manual_to_lang'] = LANGUAGES.get(manual_to_lang_display, 'zh-tw')

        # 3. 儲存引擎設定
        self.settings['selected_engine'] = self.engine_combo.get()

        # 5. 儲存常規設定
        self.settings['hotkey'] = [key.strip() for key in self.hotkey_entry.get().split('+')]
        try:
            self.settings['max_length'] = int(self.max_length_entry.get())
        except ValueError:
            self.settings['max_length'] = 500  # Fallback to default

        save_settings(self.settings)

        # 透過回呼觸發主視窗更新
        if self.on_save_callback:
            self.master.after(10, self.on_save_callback)
        self.destroy()

class ScreenshotWindow(ctk.CTkToplevel):
    def __init__(self, master):
        super().__init__(master)
        self.master = master
        self.withdraw() # 初始隱藏視窗

        self.virtual_screen_x = 0
        self.virtual_screen_y = 0
        self.virtual_screen_width = self.winfo_screenwidth() # Default to primary screen width
        self.virtual_screen_height = self.winfo_screenheight() # Default to primary screen height

        if platform.system() == "Windows":
            try:
                # Constants for GetSystemMetrics
                SM_XVIRTUALSCREEN = 76
                SM_YVIRTUALSCREEN = 77
                SM_CXVIRTUALSCREEN = 78
                SM_CYVIRTUALSCREEN = 79

                user32 = ctypes.windll.user32
                # DPI awareness should be set globally at app start, so GetSystemMetrics should return scaled pixels
                self.virtual_screen_x = user32.GetSystemMetrics(SM_XVIRTUALSCREEN)
                self.virtual_screen_y = user32.GetSystemMetrics(SM_YVIRTUALSCREEN)
                self.virtual_screen_width = user32.GetSystemMetrics(SM_CXVIRTUALSCREEN)
                self.virtual_screen_height = user32.GetSystemMetrics(SM_CYVIRTUALSCREEN)

                log_message(f"Virtual screen: x={self.virtual_screen_x}, y={self.virtual_screen_y}, w={self.virtual_screen_width}, h={self.virtual_screen_height}")
                self.overrideredirect(True) # Remove window decorations, crucial for spanning
                self.geometry(f"{self.virtual_screen_width}x{self.virtual_screen_height}+{self.virtual_screen_x}+{self.virtual_screen_y}")
            except Exception as e:
                log_error(f"Error getting virtual screen metrics: {e}. Falling back to primary screen fullscreen.")
                # Fallback to old behavior if GetSystemMetrics fails
                self.attributes('-fullscreen', True)
        else:
            # Fallback for non-Windows systems
            log_message("Non-Windows system. Using primary screen fullscreen for screenshot window.")
            self.attributes('-fullscreen', True)

        self.attributes('-alpha', 0.3) # 設定半透明
        self.attributes('-topmost', True) # 保持在最上層

        self.canvas = ctk.CTkCanvas(self, cursor="cross", bg="grey")
        self.canvas.pack(fill="both", expand=True)

        self.start_x = None
        self.start_y = None
        self.rect = None

        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
        self.bind("<Escape>", self.cancel_screenshot)

    def on_button_press(self, event):
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)
        if self.rect:
            self.canvas.delete(self.rect)
        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline='red', width=2)

    def on_mouse_drag(self, event):
        cur_x, cur_y = (self.canvas.canvasx(event.x), self.canvas.canvasy(event.y))
        self.canvas.coords(self.rect, self.start_x, self.start_y, cur_x, cur_y)

    def on_button_release(self, event):
        end_x, end_y = (self.canvas.canvasx(event.x), self.canvas.canvasy(event.y))
        
        x1 = min(self.start_x, end_x)
        y1 = min(self.start_y, end_y)
        _x1 = min(self.start_x, end_x)
        _y1 = min(self.start_y, end_y)
        _x2 = max(self.start_x, end_x)
        _y2 = max(self.start_y, end_y)

        # The coordinates from canvas are relative to the ScreenshotWindow.
        # If ScreenshotWindow is positioned at (virtual_screen_x, virtual_screen_y),
        # then the absolute coordinates for ImageGrab.grab are:
        # (virtual_screen_x + _x1, virtual_screen_y + _y1, ...)
        abs_x1 = self.virtual_screen_x + _x1
        abs_y1 = self.virtual_screen_y + _y1
        abs_x2 = self.virtual_screen_x + _x2
        abs_y2 = self.virtual_screen_y + _y2
        
        log_message(f"Relative coords: ({_x1},{_y1},{_x2},{_y2}), Virt Screen Origin: ({self.virtual_screen_x},{self.virtual_screen_y})")
        log_message(f"Absolute coords for grab: ({abs_x1},{abs_y1},{abs_x2},{abs_y2})")

        self.withdraw()
        self.master.take_screenshot((int(abs_x1), int(abs_y1), int(abs_x2), int(abs_y2)))

    def cancel_screenshot(self, event=None):
        self.withdraw()
        self.master.deiconify()

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("超酷炫翻譯")
        self.geometry("700x600")
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # 載入設定
        self.settings = load_settings()
        self.hotkey_combination = self.settings.get('hotkey', ['ctrl', 'alt', 's'])
        self.listener = None
        self.hotkey_thread = None
        self.ocr_reader = None
        self.icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'icon.ico')

        # 在背景執行緒中初始化 OCR 引擎
        threading.Thread(target=self.initialize_ocr_engine, daemon=True).start()

        # 建立頂部框架
        self.top_frame = ctk.CTkFrame(self)
        self.top_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        self.top_frame.grid_columnconfigure(0, weight=1)  # 讓標籤佔滿空間

        self.hotkey_label = ctk.CTkLabel(self.top_frame, text="", font=("Arial", 12))
        self.hotkey_label.grid(row=0, column=0, padx=10, sticky="w")

        self.settings_btn = ctk.CTkButton(self.top_frame, text="⚙️ 設定", width=80, command=self.show_settings)
        self.settings_btn.grid(row=0, column=1, padx=10)

        # 主框架
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=1, column=0, padx=10, pady=0, sticky="nsew")
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(1, weight=1)
        self.main_frame.grid_rowconfigure(3, weight=1)

        # 原文區域
        original_text_header_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        original_text_header_frame.grid(row=0, column=0, padx=5, pady=(5, 2), sticky="ew")
        original_text_header_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(original_text_header_frame, text="原文", font=("Arial", 16, "bold")).grid(row=0, column=0, sticky="w")
        self.translate_btn = ctk.CTkButton(original_text_header_frame, text="翻譯", width=80, command=self.manual_translate)
        self.translate_btn.grid(row=0, column=1, sticky="e")

        self.original_textbox = ctk.CTkTextbox(self.main_frame, font=("Arial", 14), border_width=1)
        self.original_textbox.grid(row=1, column=0, padx=5, pady=(0, 10), sticky="nsew")
        self.original_textbox.insert("1.0", "在此輸入文字進行翻譯，或使用快捷鍵截圖...")

        # 翻譯結果區域
        translated_text_header_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        translated_text_header_frame.grid(row=2, column=0, padx=5, pady=(5, 2), sticky="ew")
        translated_text_header_frame.grid_columnconfigure(0, weight=1)

        left_frame = ctk.CTkFrame(translated_text_header_frame, fg_color="transparent")
        left_frame.grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(left_frame, text="翻譯結果", font=("Arial", 16, "bold")).pack(side="left")
        self.detected_lang_label = ctk.CTkLabel(left_frame, text="", font=("Arial", 12), text_color="gray")
        self.detected_lang_label.pack(side="left", padx=10)

        self.copy_btn = ctk.CTkButton(translated_text_header_frame, text="複製", width=80, command=self.copy_translated_text)
        self.copy_btn.grid(row=0, column=1, sticky="e")

        self.translated_textbox = ctk.CTkTextbox(self.main_frame, font=("Arial", 14), border_width=1, state="disabled")
        self.translated_textbox.grid(row=3, column=0, padx=5, pady=(0, 5), sticky="nsew")

        # 狀態列
        self.status_label = ctk.CTkLabel(self, text="準備就緒", anchor="w", font=("Arial", 12))
        self.status_label.grid(row=2, column=0, padx=10, pady=(5, 5), sticky="ew")

        self.screenshot_window = None

        # 延後啟動快捷鍵監聽
        self.after(100, self.setup_hotkey)

        # 設定關閉行為
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.tray_thread = None # 初始化 tray_thread 屬性
        self.setup_tray_icon() # 呼叫 setup_tray_icon，它將在內部設定 self.tray_icon 並啟動執行緒

    def setup_hotkey(self):
        """停止舊的監聽器並啟動一個新的。"""
        if self.listener:
            try:
                self.listener.stop()
            except Exception as e:
                log_message(f"停止舊的監聽器時發生錯誤: {e}")
        
        if self.hotkey_thread and self.hotkey_thread.is_alive():
            # 這裡不需要 join，因為 listener.stop() 應該已經處理了執行緒的停止
            # 如果 listener 是 pynput.keyboard.Listener，它會在 stop() 後自動結束執行緒
            pass

        self.hotkey_combination = self.settings.get('hotkey', ['ctrl', 'alt', 's'])
        self.hotkey_thread = threading.Thread(target=self.start_hotkey_listener, daemon=True)
        self.hotkey_thread.start()
        log_message(f"正在監聽快捷鍵: {'+'.join(self.hotkey_combination)}")

        hotkey_str = '+'.join(self.hotkey_combination).upper().replace("CONTROL", "CTRL")
        self.hotkey_label.configure(text=f"快捷鍵: {hotkey_str}")

    def show_settings(self):
        """顯示設定視窗"""
        if not hasattr(self, 'settings_window') or not self.settings_window.winfo_exists():
            self.settings_window = SettingsWindow(self, self.settings, self.settings_saved)
        self.settings_window.focus()

    def manual_translate(self):
        """手動觸發翻譯原文框中的文字"""
        original_text = self.original_textbox.get("1.0", "end-1c").strip()
        if not original_text or original_text == "在此輸入文字進行翻譯，或使用快捷鍵截圖...":
            self.status_label.configure(text="請先輸入文字")
            return

        self.translated_textbox.configure(state="normal")
        self.translated_textbox.delete("1.0", "end")
        self.translated_textbox.configure(state="disabled")
        self.status_label.configure(text="正在翻譯...")
        self.update_idletasks()

        to_lang = self.settings.get('manual_to_lang', 'zh-tw')
        # 手動翻譯不進行語言偵測，來源語言設為 'auto'
        threading.Thread(target=self._execute_translation, args=(original_text, to_lang, 'auto'), daemon=True).start()

    def _execute_translation(self, text, to_lang, from_lang='auto'):
        """在背景執行緒中執行翻譯並安全地更新 UI"""
        def update_ui(func, *args, **kwargs):
            if self.winfo_exists():
                self.after(0, lambda: func(*args, **kwargs))

        try:
            selected_engine_name = self.settings.get('selected_engine', 'Default')
            
            if selected_engine_name == 'Default':
                # 使用內建的 translate 函式庫
                update_ui(self.status_label.configure, text=f"正在使用預設引擎翻譯...")
                # from_lang 參數現在由函式呼叫端傳入
                translator = TransEngine(to_lang=to_lang, from_lang=from_lang)
                translated_text = translator.translate(text)
            else:
                # 使用選擇的 LLM 引擎
                llm_engine = next((e for e in self.settings.get('llm_engines', []) if e['name'] == selected_engine_name), None)
                if not llm_engine:
                    error_msg = f"錯誤: 找不到引擎 '{selected_engine_name}' 的設定。"
                    update_ui(self.status_label.configure, text=error_msg)
                    update_ui(self.translated_textbox.configure, state="normal")
                    update_ui(self.translated_textbox.delete, "1.0", "end")
                    update_ui(self.translated_textbox.insert, "1.0", error_msg)
                    update_ui(self.translated_textbox.configure, state="disabled")
                    return

                api_key = llm_engine.get('api_key', '')  # Ollama 可能不需要 API 金鑰
                model_id = llm_engine.get('model_id', '')
                provider = llm_engine.get('provider') # 先嘗試取得 provider，不設預設值
                if not provider:
                    error_msg = f"錯誤: 引擎 '{selected_engine_name}' 設定不完整，缺少 'provider' 資訊。"
                    log_error(f"{error_msg} Engine data: {llm_engine}") # 記錄有問題的引擎資料
                    update_ui(self.status_label.configure, text=error_msg)
                    update_ui(self.translated_textbox.configure, state="normal")
                    update_ui(self.translated_textbox.delete, "1.0", "end")
                    update_ui(self.translated_textbox.insert, "1.0", error_msg)
                    update_ui(self.translated_textbox.configure, state="disabled")
                    return

                # Check for placeholder or empty API key for non-Ollama providers
                if provider.lower() != 'ollama' and (not api_key or "YOUR_" in api_key.upper()):
                    error_msg = f"錯誤: {provider.upper()} API 金鑰未設定或無效。\n請前往「設定」>「LLM 引擎管理」設定有效的 API 金鑰。"
                    log_error(f"Invalid API key for {provider.upper()}: {api_key}")
                    update_ui(self.status_label.configure, text="API 金鑰無效")
                    update_ui(self.translated_textbox.configure, state="normal")
                    update_ui(self.translated_textbox.delete, "1.0", "end")
                    update_ui(self.translated_textbox.insert, "1.0", error_msg)
                    update_ui(self.translated_textbox.configure, state="disabled")
                    return

                update_ui(self.status_label.configure, text=f"正在使用 {provider.upper()} 翻譯...")
                translated_text = translate_with_llm(
                    api_key=api_key,
                    model_id=model_id,
                    provider=provider,
                    text=text,
                    from_lang=from_lang,
                    to_lang=to_lang
                )

            update_ui(self.translated_textbox.configure, state="normal")
            update_ui(self.translated_textbox.delete, "1.0", "end")
            update_ui(self.translated_textbox.insert, "1.0", translated_text)
            update_ui(self.translated_textbox.configure, state="disabled")
            to_lang_name = REVERSE_LANGUAGES.get(to_lang, to_lang)
            update_ui(self.status_label.configure, text=f"翻譯完成 ({to_lang_name}) - 使用 {selected_engine_name}")

        except Exception as e:
            log_error(e)
            error_message = f"翻譯時發生錯誤: {str(e)[:200]}"
            update_ui(self.translated_textbox.configure, state="normal")
            update_ui(self.translated_textbox.delete, "1.0", "end")
            update_ui(self.translated_textbox.insert, "1.0", error_message)
            update_ui(self.translated_textbox.configure, state="disabled")
            update_ui(self.status_label.configure, text="翻譯錯誤")

    def copy_translated_text(self):
        """複製翻譯結果到剪貼簿"""
        try:
            translated_text = self.translated_textbox.get("1.0", "end-1c").strip()
            if translated_text:
                self.clipboard_clear()
                self.clipboard_append(translated_text)
                self.status_label.configure(text="已複製到剪貼簿！")
                self.after(2000, lambda: self.status_label.configure(text="準備就緒"))
            else:
                self.status_label.configure(text="沒有可複製的內容")
        except Exception as e:
            log_error(f"複製時發生錯誤: {e}")
            self.status_label.configure(text="複製失敗")

    def settings_saved(self):
        """設定儲存後的回呼函式"""
        self.settings = load_settings()
        self.setup_hotkey()
        
        new_ocr_lang = self.settings.get('screenshot_ocr_lang')
        # 檢查 OCR 引擎是否需要重新初始化
        if self.ocr_reader and hasattr(self.ocr_reader, 'lang_list') and self.ocr_reader.lang_list:
            current_lang = self.ocr_reader.lang_list[0]
        else:
            current_lang = None

        if current_lang != new_ocr_lang:
            log_message(f"OCR 語言變更，從 {current_lang} 到 {new_ocr_lang}。正在重新初始化...")
            threading.Thread(target=self.initialize_ocr_engine, daemon=True).start()

    def initialize_ocr_engine(self):
        """根據設定初始化或重新初始化 EasyOCR 引擎，並安全地更新 UI"""
        def safe_update_status(text):
            # 這個函式現在將從背景執行緒被呼叫，所以它必須使用 self.after
            # 來將 UI 更新操作排程到主執行緒。
            if hasattr(self, 'status_label'): # 檢查 status_label 是否存在
                self.after(0, lambda: self.status_label.configure(text=text) if self.winfo_exists() else None)

        try:
            safe_update_status("正在初始化 OCR 引擎...")
            lang_code = self.settings.get('screenshot_ocr_lang', 'auto')

            if lang_code == 'auto':
                # 對於自動偵測，載入所有支援的語言。
                lang_list = [code for code in OCR_LANGUAGES.values() if code != 'auto']
                log_message(f"OCR 自動偵測模式，載入語言: {lang_list}")
            elif lang_code == 'ja':
                # 如果選擇的 OCR 語言是日文 ('ja')，則同時加入英文 ('en') 以增強對混合文本的辨識能力
                lang_list = ['ja', 'en']
                log_message("OCR 日文模式，同時載入英文以增強辨識。")
            else:
                lang_list = [lang_code]
                log_message(f"OCR 單一語言模式，載入語言: {lang_code}")

            use_gpu = torch.cuda.is_available()
            log_message(f"正在使用語言 '{lang_list}' 初始化 EasyOCR... (GPU: {use_gpu})")
            self.ocr_reader = easyocr.Reader(lang_list, gpu=use_gpu)
            log_message("EasyOCR 初始化成功。")
            safe_update_status("準備就緒")
        except Exception as e:
            log_error(e)
            error_msg = f"OCR 引擎初始化失敗: {str(e)[:100]}"
            safe_update_status(error_msg)

    def start_screenshot_flow(self):
        """啟動截圖流程"""
        self.withdraw()
        self.after(200, self.launch_screenshot_window)
    def launch_screenshot_window(self):
        """顯示截圖視窗"""
        if self.screenshot_window is None or not self.screenshot_window.winfo_exists():
            self.screenshot_window = ScreenshotWindow(self)
        self.screenshot_window.deiconify()
        self.screenshot_window.attributes('-topmost', True)

    def take_screenshot(self, bbox):
        """接收截圖座標並啟動 OCR 翻譯流程"""
        if not bbox or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            self.deiconify() # 如果截圖無效，恢復主視窗
            return
        
        self.original_textbox.delete("1.0", "end")
        self.translated_textbox.configure(state="normal")
        self.translated_textbox.delete("1.0", "end")
        self.translated_textbox.configure(state="disabled")
        self.status_label.configure(text="正在識別文字...")
        self.update_idletasks()
        
        threading.Thread(target=self._execute_ocr_and_translation, args=(bbox,), daemon=True).start()

    def _execute_ocr_and_translation(self, bbox):
        """在背景執行緒中執行 OCR 和翻譯，並安全地更新 UI"""
        def update_ui(func, *args, **kwargs):
            # 確保在 UI 執行緒中更新，且視窗存在
            if self.winfo_exists():
                self.after(0, lambda: func(*args, **kwargs))

        try:
            # 為了截圖，主視窗需要先出現一下再隱藏，否則某些系統上 grab 會失敗
            update_ui(self.deiconify)
            update_ui(self.withdraw)
            time.sleep(0.3)  # 等待視窗動畫完成

            from PIL import __version__ as PILLOW_VERSION
            pillow_version_tuple = tuple(map(int, PILLOW_VERSION.split('.')))
            if pillow_version_tuple >= (9, 3, 0):
                log_message(f"Pillow version {PILLOW_VERSION} >= 9.3.0. Using ImageGrab.grab with all_screens=True.")
                screenshot = ImageGrab.grab(bbox=bbox, all_screens=True)
            else:
                log_message(f"Pillow version {PILLOW_VERSION} < 9.3.0. Using ImageGrab.grab without all_screens=True. Multi-monitor capture might be limited.")
                screenshot = ImageGrab.grab(bbox=bbox)

            image_np = np.array(screenshot)

            if self.ocr_reader is None:
                update_ui(self.status_label.configure, text="錯誤: OCR 引擎未初始化")
                log_message("OCR 引擎未初始化，中止辨識流程。")
                return

            log_message("準備執行 OCR readtext...")
            results = self.ocr_reader.readtext(image_np)
            log_message("OCR readtext 執行完畢。")
            
            if not results:
                update_ui(self.status_label.configure, text="未識別到文字")
                log_message("OCR 結果為空。")
                update_ui(self.deiconify)
                return

            original_text = "\n".join([res[1] for res in results])
            log_message(f"OCR 辨識出的文字: {original_text[:100]}...")
            
            # --- 語言偵測 --- #
            ocr_lang_setting = self.settings.get('screenshot_ocr_lang', 'auto')
            detected_lang_code = None
            detected_lang_name = ""

            if ocr_lang_setting == 'auto' and original_text.strip():
                try:
                    detected_lang_code = langdetect.detect(original_text)
                    detected_lang_name = REVERSE_LANGUAGES.get(detected_lang_code, f"未知 ({detected_lang_code})")
                    log_message(f"自動偵測到語言: {detected_lang_name} ({detected_lang_code})")
                    update_ui(self.detected_lang_label.configure, text=f"(偵測到: {detected_lang_name})")
                except Exception as e:
                    log_error(f"語言偵測失敗: {e}")
                    update_ui(self.detected_lang_label.configure, text="(偵測失敗)")
            else:
                update_ui(self.detected_lang_label.configure, text="") # 清除上一次的標籤

            update_ui(self.original_textbox.delete, "1.0", "end")
            update_ui(self.original_textbox.insert, "1.0", original_text)
            update_ui(self.status_label.configure, text="正在翻譯...")
            update_ui(self.deiconify)

            to_lang = self.settings.get('screenshot_to_lang', 'en')
            
            # 如果自動偵測成功，將其作為來源語言傳遞給翻譯函式
            # 注意: _execute_translation 需要被修改以接受 from_lang 參數
            from_lang = detected_lang_code if detected_lang_code else 'auto'
            self._execute_translation(original_text, to_lang, from_lang=from_lang)

        except ImportError:
            log_message("Pillow library not found. Cannot take screenshot.")
            update_ui(self.status_label.configure, text="錯誤：找不到 Pillow 函式庫")
        except Exception as e:
            log_error(f"Error in _execute_ocr_and_translation: {e}")
            error_message = f"OCR 或翻譯時發生錯誤: {str(e)[:200]}"
            update_ui(self.status_label.configure, text=error_message)
            update_ui(self.original_textbox.delete, "1.0", "end")
            update_ui(self.original_textbox.insert, "1.0", "請查看 error.log 檔案以獲取詳細資訊。")
        finally:
            # 確保主視窗最終會顯示出來
            update_ui(self.deiconify)

    def start_hotkey_listener(self):
        try:
            special_keys = {'ctrl', 'alt', 'shift', 'win', 'cmd'}
            hotkey_parts = []
            for key in self.hotkey_combination:
                key_lower = key.lower()
                if key_lower in special_keys:
                    hotkey_parts.append(f'<{key_lower}>')
                else:
                    hotkey_parts.append(key_lower)
            
            hotkey_string = '+'.join(hotkey_parts)

            def on_activate():
                log_message(f"快捷鍵 {hotkey_string} 觸發！")
                # 使用 after 確保 UI 操作在主執行緒中執行
                self.after(0, self.start_screenshot_flow)

            self.listener = keyboard.GlobalHotKeys({hotkey_string: on_activate})
            log_message(f"正在監聽快捷鍵: {hotkey_string}")
            self.listener.run()
        except Exception as e:
            # 捕獲 pynput 可能的錯誤
            error_msg = f"啟動快捷鍵監聽時發生嚴重錯誤: {e}"
            log_message(error_msg)
            log_error(e) # 記錄完整錯誤
            if hasattr(self, 'status_label') and self.winfo_exists():
                self.after(0, lambda: self.status_label.configure(text=error_msg))

    def on_close(self):
        """關閉視窗時的處理，隱藏到系統托盤"""
        self.withdraw()

    def toggle_window(self, icon=None, item=None):
        """顯示或隱藏視窗 (此方法應在主執行緒中執行)"""
        if self.state() == 'withdrawn':
            self.deiconify()
            self.attributes('-topmost', True) # 嘗試將視窗置頂
            self.focus_force()          # 強制獲取焦點
            self.attributes('-topmost', False) # 取消置頂，允許其他視窗覆蓋
        else:
            self.withdraw()

    def setup_tray_icon(self):
        """在背景執行緒中設定並執行系統托盤圖示"""
        # 確保之前的 tray_icon 執行緒已停止
        if hasattr(self, 'tray_thread') and self.tray_thread and self.tray_thread.is_alive():
            if hasattr(self, 'tray_icon') and self.tray_icon:
                try:
                    self.tray_icon.stop()
                except Exception as e:
                    log_message(f"停止舊的托盤圖示時發生錯誤: {e}")
            self.tray_thread.join(timeout=1.0) # 等待執行緒結束

        def _run_tray():
            # nonlocal self # 在 Python 3 中，巢狀函式可以直接存取外部函式的變數，如果 self 是 App 的實例方法，則 self 已經在作用域內
            try:
                try:
                    image = Image.open(ICON_FILE)
                except FileNotFoundError:
                    log_message(f"找不到圖示檔案: {ICON_FILE}。正在建立預設圖示。")
                    image = Image.new('RGB', (64, 64), color='black')
                    draw = ImageDraw.Draw(image)
                    try:
                        font = ImageFont.truetype("msyh.ttc", 48) # Windows 中文字型
                    except IOError:
                        font = ImageFont.load_default() # 預設字型
                    draw.text((18, 5), "譯", font=font, fill="white") # 使用中文 '譯'
                
                menu = pystray.Menu(
                    pystray.MenuItem("顯示/隱藏", self.toggle_window_safe, default=True),
                    pystray.MenuItem("退出", self.quit_app_safe)
                )
                
                self.tray_icon = pystray.Icon("SuperCoolTranslator", image, "Super Cool Translator", menu)
                log_message("系統托盤圖示準備執行...")
                self.tray_icon.run() # 這會阻塞執行緒直到托盤圖示停止
                log_message("系統托盤圖示執行緒已結束。")
            except Exception as e:
                log_error(e)
                log_message(f"設定或執行托盤圖示時發生嚴重錯誤: {e}")
            finally:
                log_message("_run_tray 執行緒結束")

        self.tray_thread = threading.Thread(target=_run_tray, daemon=True)
        self.tray_thread.start()
        log_message("系統托盤圖示執行緒已啟動。")

    def toggle_window_safe(self):
        # 從背景執行緒安全地呼叫以顯示/隱藏視窗
        if self.winfo_exists(): # 檢查視窗是否存在
            self.after(0, self.toggle_window)
        else:
            log_message("toggle_window_safe: 視窗不存在，無法切換。")

    def quit_app_safe(self):
        # 從背景執行緒安全地退出應用程式
        log_message("quit_app_safe 被呼叫")
        if self.winfo_exists(): # 檢查視窗是否存在
            self.after(0, self.quit_app)
        else:
            log_message("quit_app_safe: 視窗不存在，直接嘗試停止托盤並退出。")
            if hasattr(self, 'tray_icon') and self.tray_icon:
                try:
                    self.tray_icon.stop()
                except Exception as e:
                    log_message(f"quit_app_safe 中停止托盤圖示時發生錯誤: {e}")
            sys.exit() # 如果視窗不存在，直接退出程序

    def quit_app(self, icon=None, item=None):
        """完全退出應用程式"""
        log_message("正在退出應用程式...")
        if self.listener:
            try:
                self.listener.stop()
            except Exception as e:
                log_message(f"停止監聽器時發生錯誤: {e}")
        
        if hasattr(self, 'tray_icon') and self.tray_icon: # 檢查 tray_icon 是否存在
            try:
                self.tray_icon.stop()
                log_message("托盤圖示已停止。")
            except Exception as e:
                log_message(f"停止托盤圖示時發生錯誤: {e}")
        
        if hasattr(self, 'tray_thread') and self.tray_thread and self.tray_thread.is_alive():
            log_message("等待托盤圖示執行緒結束...")
            self.tray_thread.join(timeout=1.0)
            log_message("托盤圖示執行緒已結束。")

        self.destroy()
        log_message("主視窗已銷毀。")

def main():
    print("--- Main function started ---")
    log_message("--- 應用程式啟動 ---")
    try:
        log_message("正在初始化 App...")
        app = App()
        log_message("App 初始化成功。")
        log_message("正在啟動主循環 (mainloop)...")
        app.mainloop()
        log_message("--- 應用程式關閉 ---")
    except Exception as e:
        log_message(f"應用程式執行期間發生未預期的錯誤: {e}")
        log_error(e) # 直接傳遞例外物件
        print(f"應用程式發生嚴重錯誤，請檢查 error.log 檔案。")
        # 在這裡可以添加一個錯誤提示視窗
        sys.exit(1)

if __name__ == "__main__":
    # 設定 high-DPI 支援
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except ImportError:
        pass # 非 Windows 系統
    except Exception as e:
        log_error(f"設定 DPI 感知時發生錯誤: {e}")
    
    main()
