param($minutes = 600)
$myshell = New-Object -com "Wscript.Shell"
Write-Host "Keeping machine awake for $minutes minutes..."
for ($i = 0; $i -lt $minutes; $i++) {
    Start-Sleep -Seconds 60
    $myshell.sendkeys("{F15}")
}
Write-Host "Keep-awake script finished."
