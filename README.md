# Kaggle 配置

部分任务(如 house-price、spaceship-titanic)的数据需要通过 Kaggle API 下载,使用前需要先配置好 Kaggle 认证。

## 1. 获取 API Key

1. 登录 [kaggle.com](https://www.kaggle.com),进入 [Account 设置页](https://www.kaggle.com/settings/account)
2. 找到 **API** 一栏,点击 **Create New Token**,记下你的用户名和生成的 key

## 2. 写入配置文件

在 `~/.kaggle/kaggle.json`(没有就新建)写入:

```json
{"username":"你的Kaggle用户名","key":"你的key"}
```

然后收紧权限:

```bash
mkdir -p ~/.kaggle
chmod 600 ~/.kaggle/kaggle.json
```

## 3. 安装依赖并测试

```bash
pip install kaggle
kaggle competitions list
```

能正常列出比赛,就说明配置成功了。

## 4. 参加对应的比赛(仅 Kaggle 比赛类任务需要)

如果任务是某个具体 Kaggle 比赛(比如 `house-price` 对应 [house-prices-advanced-regression-techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)),第一次下载前需要先在网页上打开该比赛页面,点击 **Join Competition** 同意规则,否则下载会报 403。
