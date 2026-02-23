# 设置定时任务的 PowerShell 脚本
# 使用方法：右键点击此文件 -> 使用 PowerShell 运行

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "README 自动更新任务设置" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# 项目路径
$ProjectDir = "D:\projects\ai-collection"
$ScriptPath = Join-Path $ProjectDir "update_readme.js"
$TaskName = "AI Collection - Auto Update README"

Write-Host "`n[步骤 1/3] 检查 Node.js 环境..." -ForegroundColor Yellow

# 检查 Node.js
try {
    $nodeVersion = node --version 2>&1
    Write-Host "✓ Node.js 已安装: $nodeVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ 未找到 Node.js，请先安装 Node.js" -ForegroundColor Red
    Write-Host "  下载地址: https://nodejs.org/" -ForegroundColor Yellow
    exit 1
}

# 检查 npm 依赖
Write-Host "`n[步骤 2/3] 安装依赖..." -ForegroundColor Yellow
Push-Location $ProjectDir

try {
    if (Test-Path "package.json") {
        Write-Host "正在安装 npm 依赖..." -ForegroundColor Yellow
        npm install
        Write-Host "✓ 依赖安装成功" -ForegroundColor Green
    } else {
        Write-Host "✗ 未找到 package.json" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "✗ 依赖安装失败: $_" -ForegroundColor Red
    exit 1
} finally {
    Pop-Location
}

# 检查 API Key
Write-Host "`n[步骤 3/3] 配置 DeepSeek API..." -ForegroundColor Yellow
$envFile = Join-Path $ProjectDir ".env"

if (Test-Path $envFile) {
    Write-Host "✓ .env 配置文件已存在" -ForegroundColor Green
} else {
    Write-Host "请输入你的 DeepSeek API Key:" -ForegroundColor Cyan
    $apiKey = Read-Host -MaskInput

    if ($apiKey) {
        @"
DEEPSEEK_API_KEY=$apiKey
"@ | Out-File -FilePath $envFile -Encoding UTF8
        Write-Host "✓ API Key 已保存到 .env 文件" -ForegroundColor Green
    } else {
        Write-Host "✗ 未提供 API Key，跳过配置" -ForegroundColor Yellow
        Write-Host "请手动创建 .env 文件并添加: DEEPSEEK_API_KEY=your_key" -ForegroundColor Yellow
    }
}

# 创建或更新任务计划
Write-Host "`n[步骤 4/4] 设置定时任务..." -ForegroundColor Yellow

# 检查任务是否已存在
$taskExists = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue

if ($taskExists) {
    Write-Host "任务 '$TaskName' 已存在" -ForegroundColor Yellow
    $choice = Read-Host "是否删除并重新创建? (Y/N)"
    if ($choice -eq "Y" -or $choice -eq "y") {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Host "✓ 旧任务已删除" -ForegroundColor Green
    } else {
        Write-Host "跳过任务创建" -ForegroundColor Yellow
        exit 0
    }
}

# 创建触发器 - 每 3 天运行一次
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) -RepetitionInterval (New-TimeSpan -Days 3)

# 创建动作
$action = New-ScheduledTaskAction `
    -Execute "node" `
    -Argument "`"$ScriptPath`"" `
    -WorkingDirectory $ProjectDir

# 设置 principal - 以当前用户权限运行
$principal = New-ScheduledTaskPrincipal `
    -UserId $env:USERNAME `
    -LogonType Interactive `
    -RunLevel Highest

# 创建设置
$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -DontStopOnIdleEnd `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 5)

try {
    # 注册任务
    Register-ScheduledTask `
        -TaskName $TaskName `
        -Trigger $trigger `
        -Action $action `
        -Principal $principal `
        -Settings $settings `
        -Description "每 3 天自动更新 AI Collection 项目的 README 文档" `
        -ErrorAction Stop

    Write-Host "`n✓ 定时任务创建成功!" -ForegroundColor Green
    Write-Host "`n任务详情:" -ForegroundColor Cyan
    Write-Host "  任务名称: $TaskName" -ForegroundColor White
    Write-Host "  运行频率: 每 3 天" -ForegroundColor White
    Write-Host "  脚本路径: $ScriptPath" -ForegroundColor White
    Write-Host "  下次运行: 1 分钟后开始" -ForegroundColor White
    Write-Host "`n你可以在 '任务计划程序' 中查看和修改此任务" -ForegroundColor Yellow

} catch {
    Write-Host "`n✗ 任务创建失败: $_" -ForegroundColor Red
    exit 1
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "设置完成!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "`n手动测试脚本:" -ForegroundColor Yellow
Write-Host "  cd $ProjectDir" -ForegroundColor White
Write-Host "  npm start" -ForegroundColor White
Write-Host "`n查看任务:" -ForegroundColor Yellow
Write-Host "  taskschd.msc" -ForegroundColor White
Write-Host ""