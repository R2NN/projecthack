param(
  [string]$TaskName = "LentaPriceTagsWeeklyRetrain",
  [string]$ScriptPath = "",
  [string]$At = "03:00"
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($ScriptPath)) {
  $ScriptPath = Join-Path $PSScriptRoot "weekly_retrain.ps1"
}
if (!(Test-Path -LiteralPath $ScriptPath)) {
  throw "Weekly retrain script is missing: $ScriptPath"
}

$action = New-ScheduledTaskAction `
  -Execute "powershell.exe" `
  -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$ScriptPath`""
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Sunday -At $At
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet `
  -AllowStartIfOnBatteries `
  -DontStopIfGoingOnBatteries `
  -StartWhenAvailable `
  -ExecutionTimeLimit (New-TimeSpan -Hours 12)

$task = New-ScheduledTask -Action $action -Trigger $trigger -Principal $principal -Settings $settings
Register-ScheduledTask -TaskName $TaskName -InputObject $task -Force | Out-Null
Disable-ScheduledTask -TaskName $TaskName | Out-Null

[pscustomobject]@{
  task = $TaskName
  state = "Disabled"
  schedule = "Weekly Sunday $At"
  script = $ScriptPath
  config = if ($env:LENTA_RETRAIN_CONFIG) { $env:LENTA_RETRAIN_CONFIG } else { "runtime\retrain_config.json" }
} | ConvertTo-Json
