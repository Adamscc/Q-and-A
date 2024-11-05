# 一个基于多智能体的联网查询及本地知识图谱问答系统
## 作者：Adam Chen

## 1. 创建虚拟环境
```bash
conda create -n QandA python=3.11
conda activate QandA
```
## 2. 安装所需要的库
### 1) 安装 torch
### 2) 安装其他库
```bash
pip install -r requirements.txt
```

## 3. 本地配置neo4j

## 4. 下载所需大模型
### 1) 下载 Gemma-2-2B-Instruction (作为对话Agent，可自主替换)
```bash
huggingface-cli download --resume-download google/gemma-2-2b-it --lolac-dir model/gemma-2-2b-it
```
### 2) 下载 T5-Summary (作为总结模型，可自主替换)
```bash
huggingface-cli download --resume-download utrobinmv/t5_summary_en_ru_zh_base_2048 --lolac-dir model/t5_summary
```
### 3) 下载 BERT-Base-Chinese-NER (作为NER模型，可自主替换)
```bash
huggingface-cli download --resume-download ckiplab/bert-base-chinese-ner --lolac-dir model/bert-base-chinese-ner
```

## 5. 申请注册API
### 1) 申请 Bing API : https://www.microsoft.com/en-us/bing/apis/bing-web-search-api
### 2) 申请 neo4j 账号 : https://neo4j.com/
### 3) 申请 Firecrawl API : https://www.firecrawl.dev/app
### 4) 将申请的 API_KEYS 填写到主程序中

## 6. 启动系统
### 1) 启动 neo4j
```bash
D:
cd D:\neo4j\bin
neo4j.bat console
```
### 2) 启动主程序
```bash
conda activate QandA
python QandA.py
```

## 7. Debug
```bash
python QandA.py
```
### 在输入问题的后面加上一个 "*debug" ，会自动打印出中间过程，方便Debug

## TO DO :
### 1. 编写前端网页，部署到服务器上
### 2. 部署到手机等边缘设备上
### 3. 微调自己的模型
