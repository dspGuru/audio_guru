$ErrorActionPreference = "Stop"

# Embed C# code to handle the IStream from IMAPI2
$code = @"
using System;
using System.Runtime.InteropServices;
using System.Runtime.InteropServices.ComTypes;

[Guid("0000000c-0000-0000-C000-000000000046")]
[InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
[ComImport]
public interface IStream {
    void Read([Out, MarshalAs(UnmanagedType.LPArray, SizeParamIndex = 1)] byte[] pv, int cb, IntPtr pcbRead);
    void Write([MarshalAs(UnmanagedType.LPArray, SizeParamIndex = 1)] byte[] pv, int cb, IntPtr pcbWritten);
    void Seek(long dlibMove, int dwOrigin, IntPtr plibNewPosition);
    void SetSize(long libNewSize);
    void CopyTo(IStream pstm, long cb, IntPtr pcbRead, IntPtr pcbWritten);
    void Commit(int grfCommitFlags);
    void Revert();
    void LockRegion(long libOffset, long cb, int dwLockType);
    void UnlockRegion(long libOffset, long cb, int dwLockType);
    void Stat(out System.Runtime.InteropServices.ComTypes.STATSTG pstatstg, int grfStatFlag);
    void Clone(out IStream ppstm);
}

public static class StreamHelper {
    public static void SaveStreamToFile(object comStreamObj, string path) {
        IStream comStream = (IStream)comStreamObj;
        byte[] buffer = new byte[65536];
        IntPtr bytesReadFn = Marshal.AllocCoTaskMem(4);
        
        using (System.IO.FileStream fs = new System.IO.FileStream(path, System.IO.FileMode.Create)) {
            while (true) {
                comStream.Read(buffer, buffer.Length, bytesReadFn);
                int bytesRead = Marshal.ReadInt32(bytesReadFn);
                if (bytesRead == 0) break;
                fs.Write(buffer, 0, bytesRead);
            }
        }
        Marshal.FreeCoTaskMem(bytesReadFn);
    }
}
"@

Add-Type -TypeDefinition $code

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$isoPath = Join-Path $scriptDir "signals.iso"
$tempDir = Join-Path $scriptDir "temp_iso_build"

Write-Host "Preparing to create ISO..."

# Clean up previous run
if (Test-Path $tempDir) { Remove-Item $tempDir -Recurse -Force }
if (Test-Path $isoPath) { Remove-Item $isoPath -Force }

# Create temp dir
New-Item -Path $tempDir -ItemType Directory | Out-Null

# Copy WAV files
$wavFiles = Get-ChildItem -Path $scriptDir -Filter "*.wav"
if ($wavFiles.Count -eq 0) {
    Write-Error "No .wav files found in $scriptDir"
}

Write-Host "Copying $($wavFiles.Count) WAV files to temp directory..."
foreach ($file in $wavFiles) {
    Copy-Item -Path $file.FullName -Destination $tempDir
}

# Create ISO Image
try {
    Write-Host "Initializing IMAPI2..."
    $fsi = New-Object -ComObject IMAPI2FS.MsftFileSystemImage
    
    # 6 = Joliet (2) + ISO9660 (4)
    # 1 = ISO9660 only
    # 2 = Joliet only
    # 3 = UDF
    $fsi.FileSystemsToCreate = 1 
    
    Write-Host "Adding files to image..."
    $fsi.Root.AddTree($tempDir, $false)
    
    Write-Host "Generating filesystem structure..."
    $result = $fsi.CreateResultImage()
    
    Write-Host "Writing ISO file to $isoPath..."
    [StreamHelper]::SaveStreamToFile($result.ImageStream, $isoPath)
    
    Write-Host "Success! Created $isoPath"
}
catch {
    Write-Error "Failed to create ISO: $_"
}
finally {
    # Release COM objects to unlock files
    if ($result) { [System.Runtime.InteropServices.Marshal]::ReleaseComObject($result) | Out-Null }
    if ($fsi) { [System.Runtime.InteropServices.Marshal]::ReleaseComObject($fsi) | Out-Null }
    [System.GC]::Collect()
    [System.GC]::WaitForPendingFinalizers()

    # Cleanup temp dir
    if (Test-Path $tempDir) { 
        try { Remove-Item $tempDir -Recurse -Force -ErrorAction SilentlyContinue } catch {}
    }
}
