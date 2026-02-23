# README 自动更新任务设置指南（Node.js 版本）

## 功能说明

这个自动化系统会每 3 天使用 DeepSeek API 自动更新两个 README 文档：
- `README.md` (中文版)
- `README_EN.md` (英文版)

## 前置要求

1. **Node.js 14+** - 需要安装
2. **DeepSeek API Key** - 需要从 [DeepSeek 官网](https://platform.deepseek.com/) 获取
3. **Windows 任务计划程序** - Windows 系统自带

## 快速开始

### 方法一：自动设置（推荐）

1. 右键点击 `setup_scheduled_task.ps1`
2. 选择 "使用 PowerShell 运行"
3. 按照提示输入你的 DeepSeek API Key
4. 等待任务创建完成

### 方法二：手动设置

#### 步骤 1：安装 Node.js

下载并安装 Node.js：https://nodejs.org/

#### 步骤 2：安装依赖

```powershell
cd D:\projects\ai-collection
npm install
```

#### 步骤 3：配置 API Key

复制 `.env.example` 为 `.env`，然后编辑 `.env` 文件：

```bash
DEEPSEEK_API_KEY=your_actual_api_key_here
```

#### 步骤 4：测试脚本

```powershell
npm start
```

#### 步骤 5：创建定时任务

打开任务计划程序（`taskschd.msc`），创建新任务：

- **触发器**：每 3 天重复一次
- **操作**：
  - 程序：`node`
  - 参数：`"D:\projects\ai-collection\update_readme.js"`
  - 起始于：`D:\projects\ai-collection`
- **条件**：允许在电池供电时运行
- **设置**：如果任务失败，每 5 分钟重试 3 次

## 手动运行

随时可以手动运行更新脚本：

```powershell
cd D:\projects\ai-collection
npm start
```

或：

```powershell
node update_readme.js
```

## 卸载任务

1. 右键点击 `uninstall_scheduled_task.ps1`
2. 选择 "使用 PowerShell 运行"
3. 确认删除任务

或手动在任务计划程序中删除任务 "AI Collection - Auto Update README"

## 自定义配置

### 修改更新频率

编辑 `setup_scheduled_task.ps1` 中的触发器设置：

```powershell
# 每 3 天运行一次
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) -RepetitionInterval (New-TimeSpan -Days 3)

# 改为每天运行：
$trigger = New-ScheduledTaskTrigger -Daily -At "02:00"

# 改为每周运行：
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Sunday -At "02:00"
```

### 修改 DeepSeek 模型

编辑 `update_readme.js` 中的模型参数：

```javascript
const data = {
    model: 'deepseek-chat',  // 或 'deepseek-reasoner'
    ...
};
```

### 调整 API 参数

编辑 `update_readme.js`：

```javascript
const data = {
    model: 'deepseek-chat',
    messages: [...],
    temperature: 0.7,    // 调整创造性（0-1）
    max_tokens: 16000   // 调整最大输出长度
};
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `update_readme.js` | 主更新脚本 |
| `package.json` | Node.js 项目配置 |
| `setup_scheduled_task.ps1` | 自动设置定时任务脚本 |
| `uninstall_scheduled_task.ps1` | 卸载定时任务脚本 |
| `.env.example` | API Key 配置模板 |
| `SETUP_GUIDE.md` | 本说明文档 |

## 故障排除

### 问题 1：任务未运行

**解决方案**：
1. 打开任务计划程序 (`taskschd.msc`)
2. 查找任务 "AI Collection - Auto Update README"
3. 右键 → 启用（如果被禁用）
4. 右键 → 运行，手动测试

### 问题 2：API 调用失败

**解决方案**：
1. 检查 `.env` 文件中的 API Key 是否正确
2. 确认 API Key 有足够的额度
3. 检查网络连接
4. 手动运行脚本查看详细错误信息

### 问题 3：PowerShell 执行策略错误

**解决方案**：
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 问题 4：npm install 失败

**解决方案**：
```powershell
# 清除 npm 缓存
npm cache clean --force

# 删除 node_modules 和 package-lock.json
rm -r node_modules
rm package-lock.json

# 重新安装
npm install
```

### 问题 5：Node.js 未安装

**解决方案**：
1. 访问 https://nodejs.org/
2. 下载并安装 LTS 版本
3. 重新打开终端，运行 `node --version` 确认安装成功

## 监控日志

脚本运行时会在控制台输出详细日志，包括：
- 更新时间
- API 调用状态
- 文件更新结果

你可以将这些日志重定向到文件保存：

```powershell
npm start > update_log.txt 2>&1
```

## 注意事项

1. ⚠️ API Key 安全：不要将 `.env` 文件提交到 Git
2. ⚠️ API 额度：每次更新会消耗两次 API 调用（中文 + 英文）
3. ⚠️ 网络要求：确保服务器有稳定的网络连接
4. ⚠️ 备份：建议在自动更新前手动备份重要版本

## 获取 DeepSeek API Key

1. 访问 [DeepSeek 开放平台](https://platform.deepseek.com/)
2. 注册/登录账号
3. 进入 API 管理页面
4. 创建新的 API Key
5. 复制 API Key 并在设置时输入

## 支持

如有问题，请检查：
1. Node.js 版本是否 >= 14
2. npm 依赖是否正确安装
3. API Key 是否有效
4. 网络连接是否正常
5. Windows 任务计划程序权限是否足够

## 开发

如果你想修改代码：

```powershell
# 安装依赖
npm install

# 运行脚本
npm start
```