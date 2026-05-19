Add-Type @"
using System;
using System.Runtime.InteropServices;

public static class SleepGuard {
  [DllImport("kernel32.dll", SetLastError = true)]
  public static extern uint SetThreadExecutionState(uint esFlags);
}
"@

$ErrorActionPreference = "Stop"
[uint32]$ES_CONTINUOUS = 2147483648
[uint32]$ES_SYSTEM_REQUIRED = 1
[uint32]$ES_AWAYMODE_REQUIRED = 64
$Flags = [uint32]($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED -bor $ES_AWAYMODE_REQUIRED)

while ($true) {
  [void][SleepGuard]::SetThreadExecutionState($Flags)
  Start-Sleep -Seconds 50
}
