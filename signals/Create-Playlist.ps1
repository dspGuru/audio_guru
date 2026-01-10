$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$playlistPath = Join-Path $scriptDir "signals.wpl"

# Get WAV files
$wavFiles = Get-ChildItem -Path $scriptDir -Filter "*.wav" | Sort-Object Name

if ($wavFiles.Count -eq 0) {
    Write-Warning "No .wav files found in $scriptDir"
    exit
}

$header = @"
<?xml version="1.0"?>
<smil>
    <head>
        <meta name="Generator" content="Microsoft Windows Media Player -- 12.0.7601.17514"/>
        <meta name="ItemCount" content="$($wavFiles.Count)"/>
        <title>Audio Guru Signals</title>
    </head>
    <body>
        <seq>
"@

$footer = @"
        </seq>
    </body>
</smil>
"@

$middle = ""
foreach ($file in $wavFiles) {
    # Using relative path (just filename) is standard for playlists in same dir
    $middle += "            <media src=`"$($file.Name)`"/>`n"
}

$content = $header + "`n" + $middle + $footer
$content | Out-File -FilePath $playlistPath -Encoding UTF8

Write-Host "Created playlist at $playlistPath with $($wavFiles.Count) files."
