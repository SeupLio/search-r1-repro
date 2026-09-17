param([string]$OutDir = "E:\paper\search-r1-repro\demo\audio")

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$json = Get-Content "E:\paper\search-r1-repro\demo\narration.json" -Raw -Encoding UTF8 | ConvertFrom-Json

Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.SelectVoice("Microsoft Huihui Desktop")
$synth.Rate = 3
$synth.Volume = 100
$durations = @{}

foreach ($s in $json.sections) {
    $id = $s.id
    $wav = Join-Path $OutDir "$id.wav"
    if (Test-Path $wav) { Remove-Item $wav -Force }

    $fs = New-Object System.IO.FileStream($wav, [System.IO.FileMode]::Create, [System.IO.FileAccess]::Write)
    $synth.SetOutputToWaveStream($fs)
    $synth.Speak($s.text)
    $synth.SetOutputToNull()
    $fs.Close()

    # 时长交给 Python 从 WAV 头部精确计算，这里只回报文件大小
    $durations[$id] = (Get-Item $wav).Length
    Write-Output "$id -> $((Get-Item $wav).Length) bytes"
}

$synth.Dispose()
$durations | ConvertTo-Json | Out-File (Join-Path $OutDir "_durations.json") -Encoding utf8
