# Day 19 项目依赖管理 requirements.txt

## 今天学了什么

- 什么是项目依赖
- requirements.txt 的作用
- 如何安装项目依赖
- 如何写项目运行说明
- 如何让别人运行自己的项目

## 什么是项目依赖

项目依赖就是项目运行时需要用到的第三方库。

比如本项目用到了：

- pandas
- matplotlib
- scikit-learn
- streamlit
- joblib

## requirements.txt 的作用

requirements.txt 用来记录项目需要安装哪些库。

别人拿到项目后，可以运行：

```powershell
pip install -r requirements.txt
python train_model.py
streamlit run app.py