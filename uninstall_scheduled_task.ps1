# 卸载定时任务的 PowerShell 脚本
# 使用方法：右键点击此文件 -> 使用 PowerShell 运行

$ErrorActionPreference = "Stop"

$TaskName = "AI Collection - Auto Update README"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "卸载 README 自动更新任务" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# 检查任务是否存在
$taskExists = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue

if ($taskExists) {
    Write-Host "`n找到任务: $TaskName" -ForegroundColor Yellow
    $choice = Read-Host "确认删除? (Y/N)"

    if ($choice -eq "Y" -or $choice -eq "y") {
        try {
            Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
            Write-Host "`n✓ 任务已成功删除" -ForegroundColor Green
        } catch {
            Write-Host "`n✗ 删除任务失败: $_" -ForegroundColor Red
            exit 1
        }
    } else {
        Write-Host "已取消删除" -ForegroundColor Yellow
        exit 0
    }
} else {
    Write-Host "`n未找到任务: $TaskName" -ForegroundColor Yellow
    Write-Host "任务可能已被删除或从未创建" -ForegroundColor White
    exit 0
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "卸载完成!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""