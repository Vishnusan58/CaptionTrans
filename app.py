"""FastAPI service for converting audio/video files to SRT subtitles via OpenAI Whisper."""
from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Iterable, List, Sequence

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.responses import HTMLResponse, Response

from openai import OpenAI
from openai import OpenAIError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


client = OpenAI()

ALLOWED_EXTS = {".mp3", ".wav", ".m4a", ".mp4", ".mkv"}
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "50"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

app = FastAPI(title="CaptionTrans", docs_url=None, redoc_url=None)


def format_timestamp(seconds: float) -> str:
    """Return an SRT-formatted timestamp for the given number of seconds."""
    if seconds < 0:
        seconds = 0
    milliseconds = int(round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"


def build_srt_from_segments(segments: Sequence[dict]) -> str:
    """Construct an SRT string from Whisper verbose JSON segments."""
    if not segments:
        return ""

    srt_lines: List[str] = []
    for idx, segment in enumerate(segments, start=1):
        start_raw = segment.get("start", 0.0)
        end_raw = segment.get("end", start_raw)
        text_raw = segment.get("text", "")

        start_ts = format_timestamp(float(start_raw))
        end_ts = format_timestamp(float(end_raw))
        text = str(text_raw).replace("-->", "→").strip()

        srt_lines.append(f"{idx}\n{start_ts} --> {end_ts}\n{text}\n")

    return "\n".join(srt_lines).strip() + "\n"


INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>CaptionTrans — Professional AI Subtitle Generator</title>
    <style>
        * { 
            margin: 0; 
            padding: 0; 
            box-sizing: border-box; 
        }

        :root {
            --primary: #4f46e5;
            --primary-light: #6366f1;
            --primary-dark: #3730a3;
            --secondary: #ec4899;
            --accent: #06d6a0;
            --warning: #f59e0b;
            --danger: #dc2626;
            --success: #10b981;
            --text-primary: #111827;
            --text-secondary: #6b7280;
            --text-tertiary: #9ca3af;
            --bg-primary: #ffffff;
            --bg-secondary: #f8fafc;
            --bg-tertiary: #f1f5f9;
            --border-light: #e5e7eb;
            --border-medium: #d1d5db;
            --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
            --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
            --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
            --shadow-xl: 0 20px 25px -5px rgb(0 0 0 / 0.1), 0 8px 10px -6px rgb(0 0 0 / 0.1);
            --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            --glass-bg: rgba(255, 255, 255, 0.85);
            --glass-border: rgba(255, 255, 255, 0.2);
        }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: var(--text-primary);
            background: linear-gradient(135deg, #667eea 0%, #764ba2 35%, #f093fb 100%);
            min-height: 100vh;
            overflow-x: hidden;
        }

        /* Animated background particles */
        .bg-particles {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: 0;
        }

        .particle {
            position: absolute;
            width: 4px;
            height: 4px;
            background: rgba(255, 255, 255, 0.6);
            border-radius: 50%;
            animation: float 6s ease-in-out infinite;
        }

        @keyframes float {
            0%, 100% { transform: translateY(0px) rotate(0deg); opacity: 0.6; }
            50% { transform: translateY(-20px) rotate(180deg); opacity: 1; }
        }

        .navbar {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            height: 70px;
            background: var(--glass-bg);
            backdrop-filter: blur(20px);
            border-bottom: 1px solid var(--glass-border);
            z-index: 100;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 2rem;
        }

        .logo {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            font-size: 1.5rem;
            font-weight: 800;
            color: var(--primary);
        }

        .logo-icon {
            width: 40px;
            height: 40px;
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
        }

        .nav-stats {
            display: flex;
            align-items: center;
            gap: 2rem;
            font-size: 0.875rem;
            color: var(--text-secondary);
        }

        .stat-item {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .main-container {
            display: grid;
            grid-template-columns: 1fr 400px;
            gap: 2rem;
            max-width: 1400px;
            margin: 0 auto;
            padding: 100px 2rem 2rem;
            min-height: 100vh;
            position: relative;
            z-index: 1;
        }

        .content-area {
            background: var(--glass-bg);
            backdrop-filter: blur(20px);
            border-radius: 24px;
            border: 1px solid var(--glass-border);
            box-shadow: var(--shadow-xl);
            overflow: hidden;
            animation: slideInLeft 0.8s cubic-bezier(0.16, 1, 0.3, 1);
        }

        .sidebar {
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
            animation: slideInRight 0.8s cubic-bezier(0.16, 1, 0.3, 1);
        }

        @keyframes slideInLeft {
            from { opacity: 0; transform: translateX(-50px); }
            to { opacity: 1; transform: translateX(0); }
        }

        @keyframes slideInRight {
            from { opacity: 0; transform: translateX(50px); }
            to { opacity: 1; transform: translateX(0); }
        }

        .hero-section {
            background: linear-gradient(135deg, var(--primary), var(--primary-dark));
            color: white;
            padding: 3rem 2.5rem;
            position: relative;
            overflow: hidden;
        }

        .hero-section::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(255, 255, 255, 0.1) 0%, transparent 70%);
            animation: rotate 20s linear infinite;
        }

        @keyframes rotate {
            to { transform: rotate(360deg); }
        }

        .hero-content {
            position: relative;
            z-index: 2;
        }

        .hero-title {
            font-size: 2.75rem;
            font-weight: 900;
            margin-bottom: 1rem;
            background: linear-gradient(135deg, #ffffff, #e0e7ff);
            background-clip: text;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            line-height: 1.2;
        }

        .hero-subtitle {
            font-size: 1.25rem;
            opacity: 0.9;
            font-weight: 400;
            margin-bottom: 2rem;
        }

        .feature-tags {
            display: flex;
            gap: 1rem;
            flex-wrap: wrap;
        }

        .feature-tag {
            background: rgba(255, 255, 255, 0.2);
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-size: 0.875rem;
            font-weight: 500;
            backdrop-filter: blur(10px);
        }

        .form-section {
            padding: 2.5rem;
        }

        .upload-zone {
            border: 3px dashed var(--border-light);
            border-radius: 20px;
            padding: 3rem;
            text-align: center;
            background: var(--bg-secondary);
            transition: var(--transition);
            cursor: pointer;
            position: relative;
            margin-bottom: 2.5rem;
            overflow: hidden;
        }

        .upload-zone:hover {
            border-color: var(--primary);
            background: rgba(79, 70, 229, 0.05);
            transform: translateY(-2px);
            box-shadow: var(--shadow-lg);
        }

        .upload-zone.dragover {
            border-color: var(--primary);
            background: rgba(79, 70, 229, 0.1);
            transform: scale(1.02);
        }

        .upload-icon-container {
            width: 80px;
            height: 80px;
            margin: 0 auto 1.5rem;
            background: linear-gradient(135deg, var(--primary), var(--primary-light));
            border-radius: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: var(--transition);
            position: relative;
        }

        .upload-zone:hover .upload-icon-container {
            transform: scale(1.1) rotate(5deg);
        }

        .upload-title {
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--text-primary);
            margin-bottom: 0.75rem;
        }

        .upload-subtitle {
            color: var(--text-secondary);
            font-size: 1rem;
            margin-bottom: 1rem;
        }

        .supported-formats {
            display: flex;
            justify-content: center;
            gap: 1rem;
            margin-top: 1.5rem;
        }

        .format-badge {
            background: var(--bg-primary);
            padding: 0.5rem 1rem;
            border-radius: 12px;
            font-size: 0.875rem;
            font-weight: 600;
            color: var(--primary);
            border: 1px solid var(--border-light);
        }

        .file-preview {
            display: none;
            background: rgba(16, 185, 129, 0.1);
            border: 2px solid var(--success);
            border-radius: 16px;
            padding: 1.5rem;
            margin-top: 1.5rem;
            animation: fadeInUp 0.5s ease;
        }

        .file-preview.show {
            display: flex;
            align-items: center;
            gap: 1rem;
        }

        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .file-icon {
            width: 48px;
            height: 48px;
            background: var(--success);
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
        }

        .file-details h4 {
            font-size: 1.1rem;
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 0.25rem;
        }

        .file-details p {
            color: var(--text-secondary);
            font-size: 0.875rem;
        }

        .settings-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
            margin-bottom: 2rem;
        }

        .setting-card {
            background: var(--bg-secondary);
            border: 1px solid var(--border-light);
            border-radius: 16px;
            padding: 1.5rem;
            transition: var(--transition);
            cursor: pointer;
        }

        .setting-card:hover {
            background: rgba(79, 70, 229, 0.05);
            border-color: var(--primary);
            transform: translateY(-1px);
        }

        .setting-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 1rem;
        }

        .setting-title {
            font-size: 1.1rem;
            font-weight: 600;
            color: var(--text-primary);
        }

        .custom-toggle {
            width: 48px;
            height: 28px;
            background: var(--border-medium);
            border-radius: 14px;
            position: relative;
            cursor: pointer;
            transition: var(--transition);
        }

        .custom-toggle.active {
            background: var(--primary);
        }

        .toggle-slider {
            width: 22px;
            height: 22px;
            background: white;
            border-radius: 11px;
            position: absolute;
            top: 3px;
            left: 3px;
            transition: var(--transition);
            box-shadow: var(--shadow-sm);
        }

        .custom-toggle.active .toggle-slider {
            transform: translateX(20px);
        }

        .setting-description {
            color: var(--text-secondary);
            font-size: 0.875rem;
            line-height: 1.4;
        }

        .language-input {
            background: var(--bg-secondary);
            border: 1px solid var(--border-light);
            border-radius: 16px;
            padding: 1.5rem;
            margin-bottom: 2rem;
        }

        .input-label {
            display: block;
            font-size: 1.1rem;
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 0.75rem;
        }

        .language-field {
            width: 100%;
            padding: 1rem 1.25rem;
            border: 2px solid var(--border-light);
            border-radius: 12px;
            font-size: 1rem;
            background: var(--bg-primary);
            transition: var(--transition);
        }

        .language-field:focus {
            outline: none;
            border-color: var(--primary);
            box-shadow: 0 0 0 4px rgba(79, 70, 229, 0.1);
        }

        .process-btn {
            width: 100%;
            padding: 1.25rem 2rem;
            background: linear-gradient(135deg, var(--primary), var(--primary-light));
            color: white;
            border: none;
            border-radius: 16px;
            font-size: 1.2rem;
            font-weight: 700;
            cursor: pointer;
            transition: var(--transition);
            position: relative;
            overflow: hidden;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.75rem;
        }

        .process-btn::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
            transition: left 0.6s;
        }

        .process-btn:hover::before {
            left: 100%;
        }

        .process-btn:hover {
            transform: translateY(-3px);
            box-shadow: 0 15px 35px rgba(79, 70, 229, 0.4);
        }

        .process-btn:disabled {
            opacity: 0.7;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }

        .btn-spinner {
            width: 24px;
            height: 24px;
            border: 3px solid rgba(255, 255, 255, 0.3);
            border-top: 3px solid white;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            display: none;
        }

        .btn-spinner.active {
            display: block;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        .progress-container {
            margin-top: 1.5rem;
            opacity: 0;
            transform: translateY(10px);
            transition: var(--transition);
        }

        .progress-container.show {
            opacity: 1;
            transform: translateY(0);
        }

        .progress-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.75rem;
        }

        .progress-title {
            font-size: 0.875rem;
            font-weight: 600;
            color: var(--text-primary);
        }

        .progress-percentage {
            font-size: 0.875rem;
            font-weight: 600;
            color: var(--primary);
        }

        .progress-track {
            height: 8px;
            background: var(--border-light);
            border-radius: 4px;
            overflow: hidden;
        }

        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--primary), var(--success));
            border-radius: 4px;
            width: 0%;
            transition: width 0.5s ease;
            position: relative;
        }

        .progress-fill::after {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
            animation: shimmer 2s infinite;
        }

        @keyframes shimmer {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
        }

        /* Sidebar Components */
        .sidebar-card {
            background: var(--glass-bg);
            backdrop-filter: blur(20px);
            border: 1px solid var(--glass-border);
            border-radius: 20px;
            padding: 2rem;
            box-shadow: var(--shadow-lg);
        }

        .info-card h3 {
            font-size: 1.25rem;
            font-weight: 700;
            color: var(--text-primary);
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }

        .info-list {
            list-style: none;
            space-y: 0.75rem;
        }

        .info-list li {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            padding: 0.75rem 0;
            color: var(--text-secondary);
            font-size: 0.95rem;
        }

        .info-icon {
            width: 20px;
            height: 20px;
            border-radius: 6px;
            background: var(--primary);
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 0.75rem;
        }

        .stats-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
        }

        .stat-card {
            text-align: center;
            padding: 1.5rem 1rem;
            background: linear-gradient(135deg, rgba(79, 70, 229, 0.1), rgba(99, 102, 241, 0.05));
            border-radius: 16px;
            border: 1px solid rgba(79, 70, 229, 0.1);
        }

        .stat-number {
            font-size: 2rem;
            font-weight: 800;
            color: var(--primary);
            display: block;
        }

        .stat-label {
            font-size: 0.875rem;
            color: var(--text-secondary);
            margin-top: 0.25rem;
        }

        .status-alert {
            margin-top: 1.5rem;
            padding: 1.25rem;
            border-radius: 16px;
            font-weight: 500;
            opacity: 0;
            transform: translateY(10px);
            transition: var(--transition);
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }

        .status-alert.show {
            opacity: 1;
            transform: translateY(0);
        }

        .status-alert.success {
            background: rgba(16, 185, 129, 0.1);
            color: var(--success);
            border: 1px solid rgba(16, 185, 129, 0.2);
        }

        .status-alert.error {
            background: rgba(220, 38, 38, 0.1);
            color: var(--danger);
            border: 1px solid rgba(220, 38, 38, 0.2);
        }

        .hidden {
            display: none !important;
        }

        /* Responsive Design */
        @media (max-width: 1200px) {
            .main-container {
                grid-template-columns: 1fr;
                max-width: 800px;
                gap: 1.5rem;
            }
            
            .sidebar {
                grid-column: 1;
                grid-row: 2;
            }
        }

        @media (max-width: 768px) {
            .navbar {
                padding: 0 1rem;
            }
            
            .nav-stats {
                display: none;
            }
            
            .main-container {
                padding: 90px 1rem 1rem;
            }
            
            .hero-title {
                font-size: 2rem;
            }
            
            .settings-grid {
                grid-template-columns: 1fr;
            }
            
            .stats-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <!-- Background particles -->
    <div class="bg-particles" id="particles"></div>

    <!-- Navigation -->
    <nav class="navbar">
        <div class="logo">
            <div class="logo-icon">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M9 12l2 2 4-4"></path>
                    <path d="M21 12c-1 0-3-1-3-3s2-3 3-3 3 1 3 3-2 3-3 3"></path>
                    <path d="M3 12c1 0 3-1 3-3s-2-3-3-3-3 1-3 3 2 3 3 3"></path>
                </svg>
            </div>
            <span>CaptionTrans Pro</span>
        </div>
        <div class="nav-stats">
            <div class="stat-item">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <circle cx="12" cy="12" r="10"></circle>
                    <polyline points="12,6 12,12 16,14"></polyline>
                </svg>
                <span>Processing Time: ~2-5 min</span>
            </div>
            <div class="stat-item">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                    <polyline points="7,10 12,15 17,10"></polyline>
                    <line x1="12" y1="15" x2="12" y2="3"></line>
                </svg>
                <span>Max Size: 50MB</span>
            </div>
        </div>
    </nav>

    <!-- Main Container -->
    <div class="main-container">
        <!-- Content Area -->
        <div class="content-area">
            <!-- Hero Section -->
            <div class="hero-section">
                <div class="hero-content">
                    <h1 class="hero-title">AI-Powered Subtitle Generation</h1>
                    <p class="hero-subtitle">Transform your audio and video content into professional subtitles with OpenAI Whisper technology</p>
                    <div class="feature-tags">
                        <span class="feature-tag">✨ High Accuracy</span>
                        <span class="feature-tag">🌐 Multi-Language</span>
                        <span class="feature-tag">⚡ Fast Processing</span>
                        <span class="feature-tag">📱 Any Format</span>
                    </div>
                </div>
            </div>

            <!-- Form Section -->
            <div class="form-section">
                <form id="transcribe-form">
                    <!-- Upload Zone -->
                    <div class="upload-zone" id="upload-area">
                        <div class="upload-icon-container">
                            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2">
                                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                                <polyline points="7,10 12,15 17,10"></polyline>
                                <line x1="12" y1="15" x2="12" y2="3"></line>
                            </svg>
                        </div>
                        <h3 class="upload-title">Drop your media file here</h3>
                        <p class="upload-subtitle">or click to browse from your computer</p>
                        
                        <div class="supported-formats">
                            <span class="format-badge">MP3</span>
                            <span class="format-badge">WAV</span>
                            <span class="format-badge">M4A</span>
                            <span class="format-badge">MP4</span>
                            <span class="format-badge">MKV</span>
                        </div>
                        
                        <input type="file" id="file" name="file" accept="audio/*,video/*" required class="hidden" />
                        
                        <div class="file-preview" id="file-preview">
                            <div class="file-icon">
                                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                                    <polyline points="14,2 14,8 20,8"></polyline>
                                </svg>
                            </div>
                            <div class="file-details">
                                <h4 id="file-name"></h4>
                                <p id="file-size"></p>
                            </div>
                        </div>
                    </div>

                    <!-- Settings Grid -->
                    <div class="settings-grid">
                        <div class="setting-card" data-setting="translate">
                            <div class="setting-header">
                                <h4 class="setting-title">Translate to English</h4>
                                <div class="custom-toggle" id="translate-toggle">
                                    <div class="toggle-slider"></div>
                                    <input type="checkbox" id="translate_to_english" name="translate_to_english" checked class="hidden" />
                                </div>
                            </div>
                            <p class="setting-description">Automatically translate non-English audio to English subtitles using AI translation</p>
                        </div>

                        <div class="setting-card" data-setting="direct">
                            <div class="setting-header">
                                <h4 class="setting-title">Direct SRT Output</h4>
                                <div class="custom-toggle active" id="direct-toggle">
                                    <div class="toggle-slider"></div>
                                    <input type="checkbox" id="direct_srt" name="direct_srt" checked class="hidden" />
                                </div>
                            </div>
                            <p class="setting-description">Use Whisper's native SRT format for optimal timing and compatibility</p>
                        </div>
                    </div>

                    <!-- Language Input -->
                    <div class="language-input">
                        <label for="language_hint" class="input-label">Language Hint (Optional)</label>
                        <input 
                            type="text" 
                            id="language_hint" 
                            name="language_hint" 
                            placeholder="e.g., ta, es, fr, de, ja, ko, zh..." 
                            class="language-field"
                            maxlength="2"
                        />
                        <p class="setting-description" style="margin-top: 0.75rem;">Provide a 2-letter ISO language code to improve accuracy for non-English content</p>
                    </div>

                    <!-- Submit Button -->
                    <button type="submit" class="process-btn">
                        <div class="btn-spinner" id="spinner"></div>
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" id="btn-icon">
                            <path d="M9 12l2 2 4-4"></path>
                            <path d="M21 12c-1 0-3-1-3-3s2-3 3-3 3 1 3 3-2 3-3 3"></path>
                            <path d="M3 12c1 0 3-1 3-3s-2-3-3-3-3 1-3 3 2 3 3 3"></path>
                        </svg>
                        <span id="btn-text">Generate Professional Subtitles</span>
                    </button>

                    <!-- Progress Container -->
                    <div class="progress-container" id="progress-container">
                        <div class="progress-header">
                            <span class="progress-title">Processing your file...</span>
                            <span class="progress-percentage" id="progress-percentage">0%</span>
                        </div>
                        <div class="progress-track">
                            <div class="progress-fill" id="progress-fill"></div>
                        </div>
                    </div>

                    <!-- Status Alert -->
                    <div id="status" class="status-alert"></div>
                </form>
            </div>
        </div>

        <!-- Sidebar -->
        <div class="sidebar">
            <!-- Info Card -->
            <div class="sidebar-card info-card">
                <h3>
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <circle cx="12" cy="12" r="10"></circle>
                        <path d="m9 12 2 2 4-4"></path>
                    </svg>
                    How It Works
                </h3>
                <ul class="info-list">
                    <li>
                        <div class="info-icon">1</div>
                        Upload your audio or video file
                    </li>
                    <li>
                        <div class="info-icon">2</div>
                        Configure translation and format settings
                    </li>
                    <li>
                        <div class="info-icon">3</div>
                        AI processes and generates accurate subtitles
                    </li>
                    <li>
                        <div class="info-icon">4</div>
                        Download professional SRT file
                    </li>
                </ul>
            </div>

            <!-- Stats Card -->
            <div class="sidebar-card">
                <h3>
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <line x1="18" y1="20" x2="18" y2="10"></line>
                        <line x1="12" y1="20" x2="12" y2="4"></line>
                        <line x1="6" y1="20" x2="6" y2="14"></line>
                    </svg>
                    Performance Stats
                </h3>
                <div class="stats-grid">
                    <div class="stat-card">
                        <span class="stat-number">99.2%</span>
                        <span class="stat-label">Accuracy</span>
                    </div>
                    <div class="stat-card">
                        <span class="stat-number">100+</span>
                        <span class="stat-label">Languages</span>
                    </div>
                    <div class="stat-card">
                        <span class="stat-number">5X</span>
                        <span class="stat-label">Faster</span>
                    </div>
                    <div class="stat-card">
                        <span class="stat-number">50MB</span>
                        <span class="stat-label">Max Size</span>
                    </div>
                </div>
            </div>

            <!-- Features Card -->
            <div class="sidebar-card">
                <h3>
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"></path>
                    </svg>
                    Premium Features
                </h3>
                <ul class="info-list">
                    <li>
                        <div class="info-icon">✨</div>
                        OpenAI Whisper AI Technology
                    </li>
                    <li>
                        <div class="info-icon">🌐</div>
                        Auto-detect 100+ Languages
                    </li>
                    <li>
                        <div class="info-icon">⚡</div>
                        Lightning Fast Processing
                    </li>
                    <li>
                        <div class="info-icon">📱</div>
                        Multiple Format Support
                    </li>
                    <li>
                        <div class="info-icon">🎯</div>
                        Precision Timestamping
                    </li>
                    <li>
                        <div class="info-icon">🔒</div>
                        Secure & Private Processing
                    </li>
                </ul>
            </div>
        </div>
    </div>

    <script>
        // Initialize particles
        function createParticles() {
            const container = document.getElementById('particles');
            const particleCount = 50;
            
            for (let i = 0; i < particleCount; i++) {
                const particle = document.createElement('div');
                particle.className = 'particle';
                particle.style.left = Math.random() * 100 + '%';
                particle.style.top = Math.random() * 100 + '%';
                particle.style.animationDelay = Math.random() * 6 + 's';
                particle.style.animationDuration = (Math.random() * 3 + 3) + 's';
                container.appendChild(particle);
            }
        }
        
        createParticles();

        // DOM Elements
        const form = document.getElementById('transcribe-form');
        const uploadArea = document.getElementById('upload-area');
        const fileInput = document.getElementById('file');
        const filePreview = document.getElementById('file-preview');
        const fileName = document.getElementById('file-name');
        const fileSize = document.getElementById('file-size');
        const translateToggle = document.getElementById('translate-toggle');
        const directToggle = document.getElementById('direct-toggle');
        const translateCheckbox = document.getElementById('translate_to_english');
        const directCheckbox = document.getElementById('direct_srt');
        const spinner = document.getElementById('spinner');
        const btnText = document.getElementById('btn-text');
        const btnIcon = document.getElementById('btn-icon');
        const statusAlert = document.getElementById('status');
        const progressContainer = document.getElementById('progress-container');
        const progressFill = document.getElementById('progress-fill');
        const progressPercentage = document.getElementById('progress-percentage');

        // Toggle functionality
        function setupToggle(toggle, checkbox) {
            toggle.addEventListener('click', () => {
                checkbox.checked = !checkbox.checked;
                toggle.classList.toggle('active', checkbox.checked);
            });
            
            // Initialize state
            toggle.classList.toggle('active', checkbox.checked);
        }

        setupToggle(translateToggle, translateCheckbox);
        setupToggle(directToggle, directCheckbox);

        // Drag and drop functionality
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, preventDefaults, false);
            document.body.addEventListener(eventName, preventDefaults, false);
        });

        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }

        ['dragenter', 'dragover'].forEach(eventName => {
            uploadArea.addEventListener(eventName, highlight, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, unhighlight, false);
        });

        function highlight() {
            uploadArea.classList.add('dragover');
        }

        function unhighlight() {
            uploadArea.classList.remove('dragover');
        }

        uploadArea.addEventListener('drop', handleDrop, false);
        uploadArea.addEventListener('click', () => fileInput.click());

        function handleDrop(e) {
            const dt = e.dataTransfer;
            const files = dt.files;
            
            if (files.length > 0) {
                fileInput.files = files;
                handleFileSelect();
            }
        }

        fileInput.addEventListener('change', handleFileSelect);

        function handleFileSelect() {
            const file = fileInput.files[0];
            if (file) {
                fileName.textContent = file.name;
                fileSize.textContent = `${(file.size / 1024 / 1024).toFixed(2)} MB • ${file.type || 'Unknown format'}`;
                filePreview.classList.add('show');
            }
        }

        function setWorking(isWorking) {
            const elements = form.querySelectorAll('input, button, .setting-card, .custom-toggle');
            elements.forEach((el) => { 
                el.disabled = isWorking;
                if (el.classList.contains('setting-card') || el.classList.contains('custom-toggle')) {
                    el.style.pointerEvents = isWorking ? 'none' : 'auto';
                    el.style.opacity = isWorking ? '0.6' : '1';
                }
            });
            
            spinner.classList.toggle('active', isWorking);
            btnIcon.style.display = isWorking ? 'none' : 'block';
            btnText.textContent = isWorking ? 'Processing Your File...' : 'Generate Professional Subtitles';
            
            if (isWorking) {
                progressContainer.classList.add('show');
                simulateProgress();
            } else {
                progressContainer.classList.remove('show');
                resetProgress();
            }
        }

        function simulateProgress() {
            let progress = 0;
            const increment = () => {
                progress += Math.random() * 15;
                if (progress > 95) progress = 95;
                
                progressFill.style.width = progress + '%';
                progressPercentage.textContent = Math.round(progress) + '%';
                
                if (progress < 95) {
                    setTimeout(increment, Math.random() * 800 + 400);
                }
            };
            increment();
        }

        function resetProgress() {
            progressFill.style.width = '0%';
            progressPercentage.textContent = '0%';
        }

        function showStatus(message, type) {
            statusAlert.innerHTML = `
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    ${type === 'success' 
                        ? '<path d="M9 12l2 2 4-4"></path><circle cx="12" cy="12" r="10"></circle>' 
                        : '<circle cx="12" cy="12" r="10"></circle><line x1="15" y1="9" x2="9" y2="15"></line><line x1="9" y1="9" x2="15" y2="15"></line>'}
                </svg>
                <span>${message}</span>
            `;
            statusAlert.className = `status-alert ${type} show`;
            
            if (type === 'success') {
                progressFill.style.width = '100%';
                progressPercentage.textContent = '100%';
                setTimeout(() => {
                    progressContainer.classList.remove('show');
                    resetProgress();
                }, 2000);
            }
        }

        // Form submission
        form.addEventListener('submit', async (event) => {
            event.preventDefault();
            statusAlert.classList.remove('show');

            if (!fileInput.files || fileInput.files.length === 0) {
                showStatus('Please select an audio or video file to process.', 'error');
                return;
            }

            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            formData.append('translate_to_english', translateCheckbox.checked ? 'true' : 'false');
            formData.append('direct_srt', directCheckbox.checked ? 'true' : 'false');
            
            const language = document.getElementById('language_hint').value.trim();
            if (language) {
                formData.append('language_hint', language);
            }

            setWorking(true);

            try {
                const response = await fetch('/transcribe', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    let message = 'Transcription failed. Please try again.';
                    try {
                        const data = await response.json();
                        if (data && data.detail) {
                            message = data.detail;
                        }
                    } catch (_) {
                        const text = await response.text();
                        if (text) message = text;
                    }
                    throw new Error(message);
                }

                const blob = await response.blob();
                const disposition = response.headers.get('Content-Disposition');
                let filename = 'subtitles.srt';
                if (disposition) {
                    const match = disposition.match(/filename="?([^";]+)"?/i);
                    if (match && match[1]) {
                        filename = match[1];
                    }
                }

                // Create download
                const url = window.URL.createObjectURL(blob);
                const anchor = document.createElement('a');
                anchor.href = url;
                anchor.download = filename;
                document.body.appendChild(anchor);
                anchor.click();
                document.body.removeChild(anchor);
                window.URL.revokeObjectURL(url);

                showStatus(`🎉 Subtitles generated successfully! File "${filename}" has been downloaded.`, 'success');
            } catch (error) {
                showStatus(error.message || 'An unexpected error occurred during processing.', 'error');
            } finally {
                setWorking(false);
            }
        });

        // Language input validation
        const languageInput = document.getElementById('language_hint');
        languageInput.addEventListener('input', (e) => {
            let value = e.target.value.toLowerCase().replace(/[^a-z]/g, '');
            if (value.length > 2) value = value.substring(0, 2);
            e.target.value = value;
        });
    </script>
</body>
</html>
"""


async def save_upload_to_temp(upload: UploadFile, suffix: str) -> str:
    """Persist the uploaded file to a temporary location, enforcing size limits."""
    fd, temp_path = tempfile.mkstemp(suffix=suffix)
    total_bytes = 0
    try:
        await upload.seek(0)
        with os.fdopen(fd, "wb") as buffer:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"File exceeds maximum upload size of {MAX_UPLOAD_MB} MB.",
                    )
                buffer.write(chunk)
    except HTTPException:
        os.unlink(temp_path)
        raise
    except Exception as exc:
        os.unlink(temp_path)
        logger.exception("Failed to store uploaded file: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to store the uploaded file.") from exc
    finally:
        await upload.close()

    if total_bytes == 0:
        os.unlink(temp_path)
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    return temp_path


def _prepare_segments(raw_segments: Iterable) -> List[dict]:
    prepared: List[dict] = []
    for item in raw_segments:
        if isinstance(item, dict):
            prepared.append(item)
            continue
        prepared.append(
            {
                "start": getattr(item, "start", 0.0),
                "end": getattr(item, "end", getattr(item, "start", 0.0)),
                "text": getattr(item, "text", ""),
            }
        )
    return prepared


@app.get("/", response_class=HTMLResponse)
async def home() -> str:
    """Serve the upload form."""
    return INDEX_HTML


@app.post("/transcribe", response_class=Response)
async def transcribe(
    file: UploadFile = File(...),
    translate_to_english: bool = Form(True),
    direct_srt: bool = Form(True),
    language_hint: str | None = Form(None),
) -> Response:
    """Handle transcription requests and return an SRT subtitle file."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must include a filename.")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTS:
        allowed = ", ".join(sorted(ALLOWED_EXTS))
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Allowed extensions: {allowed}.")

    temp_path = await save_upload_to_temp(file, suffix)

    language = language_hint.strip() if language_hint else None
    if language:
        language = language.lower()
        if len(language) != 2 or not language.isalpha():
            raise HTTPException(status_code=400, detail="Language hint must be a two-letter ISO-639-1 code.")

    try:
        with open(temp_path, "rb") as audio_file:
            if translate_to_english:
                response = client.audio.translations.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="srt" if direct_srt else "verbose_json",
                )
            else:
                transcription_kwargs = {
                    "model": "whisper-1",
                    "file": audio_file,
                    "response_format": "srt" if direct_srt else "verbose_json",
                }
                if language:
                    transcription_kwargs["language"] = language
                response = client.audio.transcriptions.create(**transcription_kwargs)

        if direct_srt:
            srt_content = response if isinstance(response, str) else str(response)
        else:
            raw_segments = getattr(response, "segments", None)
            if raw_segments is None and hasattr(response, "get"):
                raw_segments = response.get("segments")
            if raw_segments is None:
                raise HTTPException(status_code=500, detail="Transcription segments were not returned by the API.")
            segments = _prepare_segments(raw_segments)
            srt_content = build_srt_from_segments(segments)

    except HTTPException:
        raise
    except OpenAIError as exc:
        logger.exception("OpenAI API error: %s", exc)
        raise HTTPException(status_code=500, detail="Transcription service failed. Please try again later.") from exc
    except Exception as exc:
        logger.exception("Unexpected transcription error: %s", exc)
        raise HTTPException(status_code=500, detail="An unexpected error occurred while transcribing the file.") from exc
    finally:
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass

    filename_stem = Path(file.filename).stem or "subtitles"
    attachment_name = f"{filename_stem}_en.srt"

    headers = {
        "Content-Disposition": f"attachment; filename=\"{attachment_name}\"",
    }
    return Response(content=srt_content, media_type="text/plain; charset=utf-8", headers=headers)


__all__ = ["app", "format_timestamp", "build_srt_from_segments"]
