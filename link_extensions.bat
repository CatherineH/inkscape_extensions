Rem merge this extensions folder with your official inkscape extensions folder by creating symbolic links to the inx and py files
Rem you may have set this differently so don't run this blindly

set ROAMING=%userprofile%\AppData\Roaming\inkscape\extensions


FORFILES /D +01/01/2001 /p %CD% /m *.py /c "cmd /c mklink /H %ROAMING%\@fname.py %CD%\@fname.py"
FORFILES /D +01/01/2001 /p %CD% /m *.inx /c "cmd /c mklink /H %ROAMING%\@fname.inx %CD%\@fname.inx"