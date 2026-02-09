$Action = New-ScheduledTaskAction -Execute 'PowerShell.exe' `
    -Argument '-NoProfile -WindowStyle Hidden -Command "Remove-Item -Path ''C:\tmp\video\*'' -Recurse -Force"'

$Trigger = New-ScheduledTaskTrigger -Daily -At 18:06

$Principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -LogonType ServiceAccount -RunLevel Highest

Register-ScheduledTask -TaskName "LimpiarDirectoriosDiarioTmpVideo" `
    -Action $Action -Trigger $Trigger -Principal $Principal `
    -Description "Limpia directorios diariamente a las 00:00"
